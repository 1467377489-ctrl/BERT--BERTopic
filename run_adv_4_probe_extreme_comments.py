# 文件名: run_adv_4_advanced_probe.py
# 功能: 智能过滤、去重、去相似后，探查两极情感得分的典型评论

import pandas as pd
import os
import re
from sentence_transformers import SentenceTransformer, util
import torch

# --- 配置区 ---
INPUT_FILE = 'results/final_data_with_aspects.xlsx'
OUTPUT_FILE = 'results/advanced_analysis/advanced_extreme_comments_report.txt'
PLATFORM_COLUMN = 'platform'
SENTIMENT_COLUMN = 'sentiment_score'
ASPECT_COLUMN = 'aspect'
COMMENT_COLUMN = 'cleaned_text'
N_COMMENTS = 20 # 每个类别展示的评论数量

# --- 智能过滤参数 ---
MIN_LENGTH = 8 # 稍微提高长度门槛，过滤更多噪音
MIN_DIVERSITY_RATIO = 0.4 # 提高多样性门槛

# --- 【核心】相似度过滤参数 ---
ENABLE_SIMILARITY_FILTER = True # 是否开启相似度过滤
SIMILARITY_THRESHOLD = 0.85 # 相似度得分高于此阈值的评论将被视为重复
SIMILARITY_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2' # 一个轻量级且高效的多语言模型

def is_meaningful(text):
    """判断一条评论是否具有初步研究意义的函数"""
    if not isinstance(text, str) or len(text) < MIN_LENGTH:
        return False
    text_for_diversity = re.sub(r':\w+:', '', text)
    if not text_for_diversity:
        return False
    num_unique_chars = len(set(text_for_diversity))
    diversity_ratio = num_unique_chars / len(text_for_diversity)
    if diversity_ratio < MIN_DIVERSITY_RATIO:
        return False
    return True

def filter_similar_comments(df_sorted):
    """
    使用句子相似度过滤掉内容高度重复的评论。
    """
    if not ENABLE_SIMILARITY_FILTER or df_sorted.empty:
        return df_sorted

    print("    > 正在进行相似度过滤...")
    # 加载模型
    model = SentenceTransformer(SIMILARITY_MODEL)
    
    # 提取文本并计算向量
    comments = df_sorted[COMMENT_COLUMN].tolist()
    embeddings = model.encode(comments, convert_to_tensor=True, show_progress_bar=False)
    
    # 计算相似度矩阵
    cosine_scores = util.cos_sim(embeddings, embeddings)
    
    # 筛选出要保留的评论的索引
    indices_to_keep = []
    indices_to_discard = set()
    
    for i in range(len(comments)):
        if i in indices_to_discard:
            continue
        indices_to_keep.append(i)
        # 将与当前评论过于相似的后续评论加入丢弃列表
        for j in range(i + 1, len(comments)):
            if cosine_scores[i][j] > SIMILARITY_THRESHOLD:
                indices_to_discard.add(j)
                
    # 返回过滤后的DataFrame
    return df_sorted.iloc[indices_to_keep]

def main():
    """主函数，智能过滤后探查两极评论"""
    print("--- 启动进阶分析4: 【高级过滤版】两极评论定性探查 ---")
    os.makedirs('results/advanced_analysis/', exist_ok=True)
    
    df = pd.read_excel(INPUT_FILE)
    
    # 1. 基础过滤
    print(f"  > 原始数据共 {len(df)} 条。")
    meaningful_mask = df[COMMENT_COLUMN].apply(is_meaningful)
    df_meaningful = df[meaningful_mask]
    print(f"  > 基础过滤后，剩余 {len(df_meaningful)} 条有意义的评论。")
    
    # 2. 精确去重
    original_count = len(df_meaningful)
    df_unique = df_meaningful.drop_duplicates(subset=[COMMENT_COLUMN])
    print(f"  > 精确去重后，剩余 {len(df_unique)} 条独一无二的评论。")
    
    report_lines = ["="*80, "                 两极情感典型评论报告 (高级过滤后)", "="*80]
    
    # 3. 分平台进行探查
    for platform in df_unique[PLATFORM_COLUMN].unique():
        report_lines.append(f"\n\n{'='*30} 平台: {platform} {'='*30}")
        df_platform = df_unique[df_unique[PLATFORM_COLUMN] == platform].copy()
        
        # 获取得分最高的评论
        report_lines.append(f"\n--- 【得分最高的 {N_COMMENTS} 条评论】 ---")
        top_comments_raw = df_platform.sort_values(SENTIMENT_COLUMN, ascending=False).head(N_COMMENTS * 3) # 多取一些用于相似度过滤
        top_comments_filtered = filter_similar_comments(top_comments_raw).head(N_COMMENTS)
        for _, row in top_comments_filtered.iterrows():
            line = f"  - 得分: {row[SENTIMENT_COLUMN]:.3f} | 方面: {row[ASPECT_COLUMN]} | 内容: {row[COMMENT_COLUMN]}"
            report_lines.append(line)
            
        # 获取得分最低的评论
        report_lines.append(f"\n--- 【得分最低的 {N_COMMENTS} 条评论】 ---")
        bottom_comments_raw = df_platform.sort_values(SENTIMENT_COLUMN, ascending=True).head(N_COMMENTS * 3)
        bottom_comments_filtered = filter_similar_comments(bottom_comments_raw).head(N_COMMENTS)
        for _, row in bottom_comments_filtered.iterrows():
            line = f"  - 得分: {row[SENTIMENT_COLUMN]:.3f} | 方面: {row[ASPECT_COLUMN]} | 内容: {row[COMMENT_COLUMN]}"
            report_lines.append(line)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
        
    print("\n--- 分析完成 ---")
    print(f"高级过滤后的典型评论报告已保存至: '{OUTPUT_FILE}'")

if __name__ == "__main__":
    main()