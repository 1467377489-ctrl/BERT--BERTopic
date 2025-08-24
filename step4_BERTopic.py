# =============================================================================
#
#         脚本 1: BERTopic 模型训练与保存 (已修正)
#
#   功能:
#   1. 加载数据并为 Bilibili 和 YouTube 平台分别训练 BERTopic 模型。
#   2. 训练成功后，保存两份产出：
#      - 主题信息的 .xlsx 文件。
#      - 完整的 BERTopic 模型对象文件夹 (用于后续可视化)。
#
# =============================================================================

import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import os
from sentence_transformers import SentenceTransformer
from bertopic.vectorizers import ClassTfidfTransformer

# --- 🚀【请在这里配置您的路径】🚀 ---
INPUT_FILE_PATH = 'data/comments_for_bertopic.xlsx'
EMBEDDING_MODEL_NAME = 'shibing624/text2vec-base-chinese'
PLATFORM_COLUMN = 'platform'
EMBEDDING_TEXT_COLUMN = 'cleaned_text_for_sentiment'
TOPIC_WORDS_COLUMN = 'text_for_bertopic'
OUTPUT_DIR_RESULTS = 'results/topic_modeling/'
# ----------------------------------------------------

# 【新增】=======================================================================
# 定义一个顶层函数来替代 lambda，以解决 PicklingError。
# Pickle 模块可以正确地序列化和反序列化这个有明确路径的函数。
def split_tokenizer(text):
    """一个简单的分词器，通过空格分割文本。"""
    return text.split()
# ===============================================================================

def main():
    print("\n--- 启动【脚本 1: 模型训练与保存】---")
    os.makedirs(OUTPUT_DIR_RESULTS, exist_ok=True)

    print(f"\n> 正在从 '{INPUT_FILE_PATH}' 加载数据...")
    try:
        df = pd.read_excel(INPUT_FILE_PATH)
    except FileNotFoundError:
        print(f"  > [严重错误] 数据文件未找到: '{INPUT_FILE_PATH}'。程序终止。")
        return
        
    df.dropna(subset=[EMBEDDING_TEXT_COLUMN, TOPIC_WORDS_COLUMN], inplace=True)
    print(f"  > 数据加载完成，共 {len(df)} 条有效评论。")

    df_bili = df[df[PLATFORM_COLUMN].str.contains('bilibili', case=False, na=False)]
    df_yt = df[df[PLATFORM_COLUMN].str.contains('youtube', case=False, na=False)]
    print(f"  > 数据拆分后: Bilibili {len(df_bili)} 条, YouTube {len(df_yt)} 条。")

    print(f"\n> 正在预加载句子模型: '{EMBEDDING_MODEL_NAME}' (可能需要一些时间)...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # --- Bilibili 平台处理 ---
    if len(df_bili) > 50:
        print("\n--- [1/2] 正在为 Bilibili 平台建模... ---")
        topic_model_bili = run_bertopic_for_platform(df_bili, embedding_model, platform="bilibili")
        if topic_model_bili:
            save_all_results(topic_model_bili, 'bilibili')
    else:
        print("\n--- [1/2] Bilibili 数据量不足 (<50条)，跳过建模。")

    # --- YouTube 平台处理 ---
    if len(df_yt) > 50:
        print("\n--- [2/2] 正在为 YouTube 平台建模... ---")
        topic_model_yt = run_bertopic_for_platform(df_yt, embedding_model, platform="youtube")
        if topic_model_yt:
            save_all_results(topic_model_yt, 'youtube')
    else:
        print("\n--- [2/2] YouTube 数据量不足 (<50条)，跳过建模。")

    print("\n--- 脚本 1 执行完毕！所有模型和结果已保存。---")

def run_bertopic_for_platform(df_platform, embedding_model_object, platform):
    docs_for_embedding = df_platform[EMBEDDING_TEXT_COLUMN].tolist()
    docs_for_topic_words = df_platform[TOPIC_WORDS_COLUMN].tolist()
    
    # 【修改处】===================================================================
    # 使用我们定义的顶层函数 split_tokenizer 替代原来的 lambda 函数。
    vectorizer_model = CountVectorizer(tokenizer=split_tokenizer, min_df=2)
    # ===============================================================================
    
    min_topic_size = 20
    ctfidf_model = ClassTfidfTransformer(bm25_weighting=True, reduce_frequent_words=True)

    topic_model = BERTopic(
        embedding_model=embedding_model_object, vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model, language="multilingual", nr_topics="auto",
        min_topic_size=min_topic_size, calculate_probabilities=True, verbose=True
    )
    
    try:
        print(f"  > 平台: {platform}, 参数: min_topic_size={min_topic_size}, BM25开启")
        embeddings = embedding_model_object.encode(docs_for_embedding, show_progress_bar=True)
        topics, probs = topic_model.fit_transform(docs_for_topic_words, embeddings)
        print(f"  > {platform} 平台建模完成，发现 {len(topic_model.get_topic_info())-1} 个主题。")
        return topic_model
    except Exception as e:
        print(f"  > [错误] 在为 {platform} 平台建模时发生错误: {e}")
        return None

def save_all_results(topic_model, platform_name):
    """统一保存Excel摘要和完整的模型对象"""
    # 1. 保存主题信息 Excel
    topic_info_df = topic_model.get_topic_info()
    excel_path = os.path.join(OUTPUT_DIR_RESULTS, f'{platform_name}_topics_info.xlsx')
    topic_info_df.to_excel(excel_path, index=False)
    print(f"  > 主题信息表已保存至: '{os.path.basename(excel_path)}'")
    
    # 2. 保存完整的模型文件夹
    model_path = os.path.join(OUTPUT_DIR_RESULTS, f'{platform_name}_bertopic_model')
    
    # 【优化处】 推荐使用 safetensors 格式保存，因为它更安全、更现代。
    # 如果失败，会自动回退到 pickle 方法。
    try:
        topic_model.save(model_path, serialization="safetensors")
        print(f"  > 完整BERTopic模型已保存至: '{os.path.basename(model_path)}' (使用 safetensors)")
    except Exception as e:
        print(f"  > [提示] 使用 safetensors 保存失败 ({e})，尝试使用 pickle 方法...")
        topic_model.save(model_path, serialization="pickle")
        print(f"  > 完整BERTopic模型已保存至: '{os.path.basename(model_path)}' (使用 pickle)")


if __name__ == "__main__":
    main()