# 文件名: step0_finetune_model.py (v3.4 - Final Production Version)

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW 
from tqdm.auto import tqdm
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ---------------- 配置区 ----------------
LABELED_FILE_PATH = 'data/test.xlsx'
MODEL_SAVE_PATH = 'models/finetuned_sentiment_model'

# 【重要】确保这里的列名与你的Excel文件和清洗脚本(preprocess_data.py)完全一致
COMMENT_COLUMN = 'cleaned_text' 
MANUAL_SCORE_COLUMN = 'manual_score'  

BASE_MODEL_NAME = 'IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment'
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LENGTH = 512 # 对于评论文本，256通常足够，可以加快训练速度
VALIDATION_SIZE = 0.1
RANDOM_SEED = 42 # 固定随机种子，保证实验可复现

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- 数据集与评估函数 (可移至 utils.py) ----------------
class CommentRegressionDataset(Dataset):
    """用于回归任务的自定义数据集"""
    def __init__(self, texts, scores, tokenizer, max_length):
        self.texts = texts
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        score = float(self.scores[idx])
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        item['labels'] = torch.tensor(score, dtype=torch.float)
        return item

def evaluate_model(model, dataloader, loss_fn, device):
    """在验证集上评估模型性能"""
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device).unsqueeze(1)
            outputs = model(**inputs)
            preds = outputs.logits
            total_loss += loss_fn(preds, labels).item()
            all_preds.extend(preds.squeeze(1).cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
    avg_loss = total_loss / len(dataloader)
    rmse = mean_squared_error(all_labels, all_preds, squared=False)
    return avg_loss, rmse

# ---------------- 主函数 ----------------
def main():
    print(f"--- 启动模型微调脚本 (设备: {DEVICE}) ---")
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    # 1. 加载数据并进行严格清洗验证
    print(f"  > 正在加载并验证标注数据: '{LABELED_FILE_PATH}'")
    try:
        df = pd.read_excel(LABELED_FILE_PATH)
        print(f"  > 原始文件加载了 {len(df)} 行。")
    except FileNotFoundError:
        print(f"错误：找不到标注文件 '{LABELED_FILE_PATH}'。请确保文件路径和名称正确。")
        return

    # 步骤 1: 检查必需的列是否存在
    required_columns = [COMMENT_COLUMN, MANUAL_SCORE_COLUMN]
    if not all(col in df.columns for col in required_columns):
        print(f"错误：标注文件中必须包含以下所有列: {required_columns}")
        print(f"  > 当前文件包含的列: {df.columns.tolist()}")
        return

    # 步骤 2: 强制转换类型并处理无效值
    df[COMMENT_COLUMN] = df[COMMENT_COLUMN].astype(str).str.strip()
    df.loc[df[COMMENT_COLUMN] == '', COMMENT_COLUMN] = None 
    df[MANUAL_SCORE_COLUMN] = pd.to_numeric(df[MANUAL_SCORE_COLUMN], errors='coerce')

    # 步骤 3: 打印出所有问题行，方便调试
    invalid_rows = df[df[COMMENT_COLUMN].isnull() | df[MANUAL_SCORE_COLUMN].isnull()]
    if not invalid_rows.empty:
        print("\n  > 警告：在数据中发现了以下问题行（文本为空或分数为非数字），它们将被丢弃：")
        with pd.option_context('display.max_rows', 10):
            print(invalid_rows)
    
    # 步骤 4: 最终丢弃所有包含任何空值的行
    original_count = len(df)
    df.dropna(subset=required_columns, inplace=True)
    final_count = len(df)
    print(f"  > 数据清洗完成。移除了 {original_count - final_count} 行无效数据。")
    
    if final_count < 20:
        print(f"错误：清洗后剩余的有效数据 ({final_count}条) 过少，无法进行训练。请检查你的标注文件。")
        return

    # 2. 划分数据集
    train_df, val_df = train_test_split(df, test_size=VALIDATION_SIZE, random_state=RANDOM_SEED)
    print(f"  > 数据集划分完成: {len(train_df)}条训练, {len(val_df)}条验证。")

    # 3. 加载模型和分词器
    print(f"  > 正在加载预训练模型: '{BASE_MODEL_NAME}'")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=1,
        ignore_mismatched_sizes=True
    ).to(DEVICE)
    print("  > 模型加载成功，分类头已重新初始化为回归任务。")

    # 4. 创建DataLoader
    train_dataset = CommentRegressionDataset(train_df[COMMENT_COLUMN].tolist(), train_df[MANUAL_SCORE_COLUMN].tolist(), tokenizer, MAX_LENGTH)
    val_dataset = CommentRegressionDataset(val_df[COMMENT_COLUMN].tolist(), val_df[MANUAL_SCORE_COLUMN].tolist(), tokenizer, MAX_LENGTH)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 5. 定义优化器和损失函数
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    loss_fn = torch.nn.MSELoss()
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    # 6. 微调训练循环
    print("--- 开始微调回归模型 ---")
    best_val_rmse = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for batch in progress_bar:
            optimizer.zero_grad()
            inputs = {k: v.to(DEVICE) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(DEVICE).unsqueeze(1)
            outputs = model(**inputs)
            preds = outputs.logits
            loss = loss_fn(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        avg_train_loss = epoch_loss / len(train_dataloader)
        
        val_loss, val_rmse = evaluate_model(model, val_dataloader, loss_fn, DEVICE)
        print(f"  Epoch {epoch+1}: Avg Train Loss = {avg_train_loss:.4f} | Val Loss = {val_loss:.4f} | Val RMSE = {val_rmse:.4f}")
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            print(f"  > 新的最优模型！(RMSE: {best_val_rmse:.4f}). 正在保存至: '{MODEL_SAVE_PATH}'")
            model.save_pretrained(MODEL_SAVE_PATH)
            tokenizer.save_pretrained(MODEL_SAVE_PATH)
    
    print("\n--- 训练完成 ---")
    print(f"最优模型已保存，其在验证集上的最佳RMSE为: {best_val_rmse:.4f}")
    print("下一步，请运行 step1_predict_scores.py 对全部数据进行预测。")

if __name__ == "__main__":
    main()