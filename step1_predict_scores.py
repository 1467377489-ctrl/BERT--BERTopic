# 文件名: step1_predict_scores.py
# 功能: 加载微调好的模型，对全部清洗后的数据进行情感分数预测。

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm
import os

# ---------------- 配置区 ----------------
# 【输入】全部已清洗的数据
INPUT_FILE_PATH = 'data/comments_for_sentiment.xlsx'
# 【输入】微调后模型的保存路径
MODEL_PATH = 'models/finetuned_sentiment_model'
# 【输出】带有情感分数的结果文件，是后续步骤的输入
OUTPUT_FILE_PATH = 'results/sentiment_scores.xlsx'

# 【列名】
COMMENT_COLUMN = 'cleaned_text'

# 【预测参数】
BATCH_SIZE = 32      # 预测时可以使用更大的batch size以提高速度
MAX_LENGTH = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- 自定义Dataset (用于预测) ----------------
class PredictionDataset(Dataset):
    """用于预测任务的自定义数据集 (无标签)"""
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {key: val.squeeze(0) for key, val in inputs.items()}

# ---------------- 主函数 ----------------
def main():
    print("--- 启动【步骤1】: 使用微调模型进行大规模预测 ---")
    os.makedirs('results', exist_ok=True)
    
    # 1. 加载全部已清洗的数据
    print(f"  > 正在加载全部数据: '{INPUT_FILE_PATH}'")
    try:
        df = pd.read_excel(INPUT_FILE_PATH)
    except FileNotFoundError:
        print(f"错误：找不到输入文件 '{INPUT_FILE_PATH}'。请先运行数据清洗脚本。")
        return

    df.dropna(subset=[COMMENT_COLUMN], inplace=True)
    texts = df[COMMENT_COLUMN].tolist()
    print(f"  > 成功加载 {len(texts)} 条评论进行预测。")

    # 2. 加载微调后的模型和分词器
    print(f"  > 正在加载微调后的模型: '{MODEL_PATH}'")
    if not os.path.exists(MODEL_PATH):
        print(f"错误：找不到微调后的模型 '{MODEL_PATH}'。请先运行 finetune_model.py 脚本。")
        return
        
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
    
    # 3. 创建DataLoader
    dataset = PredictionDataset(texts, tokenizer, MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False) # 预测时不需要打乱顺序

    # 4. 执行预测
    print("--- 开始大规模预测 ---")
    model.eval() # 切换到评估模式
    all_scores = []
    
    with torch.no_grad(): # 在预测时禁用梯度计算，节省内存和计算资源
        for batch in tqdm(dataloader, desc="Predicting"):
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}
            
            outputs = model(**inputs)
            preds = outputs.logits
            
            # .squeeze(1) 移除多余的维度，.cpu() 将数据移回CPU, .tolist() 转换为列表
            all_scores.extend(preds.squeeze(1).cpu().tolist())

    # 5. 保存预测结果
    df['sentiment_score'] = all_scores
    # 【可选】对分数进行裁剪，确保在-1到1之间
    df['sentiment_score'] = df['sentiment_score'].clip(-1, 1)
    
    df.to_excel(OUTPUT_FILE_PATH, index=False)
    print(f"\n--- 步骤1完成 ---")
    print(f"预测结果已保存至: '{OUTPUT_FILE_PATH}'")
    print("下一步，请运行 step2_aspect_analysis.py")

if __name__ == "__main__":
    main()