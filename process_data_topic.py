# 文件名: preprocess_data_for_bertopic_final.py
# 描述: 为BERTopic生成两种必需的文本：(1) 清洗过的完整句子 (2) 分词并过滤后的关键词

import pandas as pd
import re
import emoji
from opencc import OpenCC
import jieba
import jieba.posseg as pseg
import os

# --- 配置区 (与之前相同) ---
RAW_DATA_PATH = 'data/final_data.xlsx' # 假设您的原始数据在这里
# 【核心】输出文件现在包含两条管线的产出
OUTPUT_FILE_PATH = 'data/comments_for_bertopic.xlsx' 
STOPWORDS_PATH = 'stopword/hit_stopwords.txt'
COMMENT_COLUMN_NAME = 'comment_text'
ALLOWED_POS = {'n', 'nr', 'ns', 'nt', 'nz', 'v', 'vd', 'vn', 'a', 'ad', 'an'}

# --- 初始化全局组件 (与之前相同) ---
# ... (省略，请使用您之前的初始化代码)

# --- 【核心修改】将清洗流程拆分为两个函数 ---

def basic_clean(text, cc):
    """
    执行基础清洗，保留完整句子结构，用于生成语义向量。
    """
    if not isinstance(text, str) or text.strip() == '':
        return ""
    text = cc.convert(text)  # 繁转简
    text = re.sub(r'http\S+|https\S+', '', text) # 去URL
    text = re.sub(r'@\S+', '', text) # 去@
    text = re.sub(r'#\S+#', '', text) # 去话题
    text = re.sub(r'\s+', ' ', text).strip() # 规范化空格
    return text

def advanced_clean_and_tokenize(text, stopwords):
    """
    在基础清洗后，进行分词、词性过滤和停用词过滤，用于提取主题关键词。
    """
    # 移除emoji, 数字, 和标点 (这些对于关键词提取是噪音)
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'\d+', '', text)
    text = re.sub(r"[!\"#$%&'()*+,-./:;<=>?@\[\\\]^_`{|}~—，。？、！【】《》“”‘’]+", " ", text)
    
    words_with_pos = pseg.lcut(text)
    meaningful_words = [word for word, flag in words_with_pos if flag in ALLOWED_POS]
    final_words = [word for word in meaningful_words if word not in stopwords and len(word) > 1]
    
    return " ".join(final_words)

# --- 主流程 ---
def main():
    print("\n--- 启动为【BERTopic】准备数据的最终版清洗脚本 ---")
    
    # ... (加载 OpenCC, Jieba, Stopwords 的代码与您之前的一样)
    cc = OpenCC('t2s')
    jieba.initialize()
    stopwords = set() # 假设停用词加载...

    df = pd.read_excel(RAW_DATA_PATH)
    
    print("\n[步骤 1/2] 正在进行文本清洗...")
    # 【核心】生成第一列：清洗过的完整句子 (用于Embedding)
    df['cleaned_text_for_sentiment'] = df[COMMENT_COLUMN_NAME].apply(lambda x: basic_clean(x, cc))
    
    # 【核心】生成第二列：分词后的关键词 (用于Topic Words)
    # 注意：这里我们对已经基础清洗过的文本进行处理
    df['text_for_bertopic'] = df['cleaned_text_for_sentiment'].apply(lambda x: advanced_clean_and_tokenize(x, stopwords))
    
    # 移除两列都为空的行
    df.dropna(subset=['cleaned_text_for_sentiment', 'text_for_bertopic'], inplace=True)
    df = df[df['text_for_bertopic'] != '']
    
    print(f"\n[步骤 2/2] 保存最终数据至 '{OUTPUT_FILE_PATH}'...")
    # 只保留后续需要的列，让文件更干净
    final_df = df[[COMMENT_COLUMN_NAME, 'cleaned_text_for_sentiment', 'text_for_bertopic', 'platform']]
    final_df.to_excel(OUTPUT_FILE_PATH, index=False)
    
    print("\n--- 清洗流程成功结束 ---")
    print(f"最终数据已保存。现在可以使用 '{OUTPUT_FILE_PATH}' 进行BERTopic分析。")

if __name__ == "__main__":
    main()