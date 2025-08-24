# =============================================================================
#
#         脚本 2: BERTopic 独立可视化 (最终版 v2 - 对数尺寸缩放)
#
# =============================================================================
import os
import pandas as pd
from bertopic import BERTopic
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import umap
from scipy.cluster import hierarchy as sch
from sentence_transformers import SentenceTransformer
from adjustText import adjust_text
# 【新增】导入 numpy 用于对数计算
import numpy as np

# --- 🚀【请在这里配置您的路径和美学参数】🚀 ---
# 1. 模型和数据路径
RESULTS_DIR = 'results/topic_modeling/'
ORIGINAL_DATA_PATH = 'data/comments_for_bertopic.xlsx'
EMBEDDING_MODEL_NAME = 'shibing624/text2vec-base-chinese'
FONT_PATH = '/kaggle/input/chinese-font/SourceHanSansSC-Regular.otf' 
VIS_OUTPUT_DIR = 'results/visualization_final/'

# --- 🎨 1. 美化配置：定义字体和配色 ---
try:
    font_prop = fm.FontProperties(fname=FONT_PATH)
    print(f"  > 成功加载字体: {os.path.basename(FONT_PATH)}")
except Exception as e:
    print(f"!!! [严重错误] 字体文件加载失败: '{FONT_PATH}'. 错误: {e}")
    font_prop = fm.FontProperties()

UNIFIED_COLOR = "#336699"         
PALETTE_FREQ_BAR = "Blues_r"      
PALETTE_WORD_BAR = "Blues_r"      


def visualize_single_platform(platform_name, df_docs, embedding_model_object):
    """为单个平台加载模型并生成所有可视化图表"""
    model_path = os.path.join(RESULTS_DIR, f'{platform_name}_bertopic_model')
    if not os.path.exists(model_path):
        print(f"模型路径不存在，跳过平台 '{platform_name}': {model_path}")
        return

    print(f"\n--- 正在为平台 '{platform_name}' 生成可视化 ---")
    print(f"  > 正在从 '{model_path}' 加载模型...")
    topic_model = BERTopic.load(model_path, embedding_model=embedding_model_object)
    
    generate_frequency_plot(topic_model, platform_name)
    generate_barchart_plot(topic_model, platform_name) 
    generate_distance_map_plot(topic_model, platform_name) 
    generate_hierarchy_plot(topic_model, platform_name)
    print(f"  --- '{platform_name}' 的所有图表生成完毕 ---")


def generate_frequency_plot(model, platform):
    print(f"    > 正在生成 [主题频率图]...")
    # ... (此函数无变化)
    topic_info = model.get_topic_info(); freq_df = topic_info[topic_info['Topic'] != -1]
    top_n = min(15, len(freq_df)); freq_df_display = freq_df.head(top_n)
    plt.figure(figsize=(10, 8))
    b = sns.barplot(data=freq_df_display, x="Count", y="Name", palette=PALETTE_FREQ_BAR)
    b.set_title(f'{platform.capitalize()} - Top {top_n} Topic Frequency', fontproperties=font_prop, fontsize=16)
    b.set_xlabel('Document Count', fontproperties=font_prop, fontsize=14)
    b.set_ylabel('Topic', fontproperties=font_prop, fontsize=14)
    for label in b.get_yticklabels():
        label.set_fontproperties(font_prop)
    sns.despine(); plt.tight_layout()
    plt.savefig(os.path.join(VIS_OUTPUT_DIR, f"{platform}_topic_frequency.png"), bbox_inches='tight')
    plt.savefig(os.path.join(VIS_OUTPUT_DIR, f"{platform}_topic_frequency.pdf"), bbox_inches='tight')
    plt.close()


def generate_barchart_plot(model, platform):
    """为每个主题生成一个独立的、清晰的词语权重条形图"""
    print(f"    > 正在生成 [独立主题词条形图] (一图一主题)...")
    # ... (此函数无变化)
    topic_info = model.get_topic_info(); freq_df = topic_info[topic_info['Topic'] != -1]
    top_n_topics_to_plot = min(8, len(freq_df))
    topics_to_visualize = freq_df['Topic'].head(top_n_topics_to_plot)
    for topic_id in topics_to_visualize:
        words_scores = model.get_topic(topic_id)
        if not words_scores: continue
        topic_name = model.get_topic_info(topic_id)['Name'].iloc[0]
        words = [w for w, s in words_scores]; scores = [s for w, s in words_scores]
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = sns.color_palette(PALETTE_WORD_BAR, len(words))
        ax.barh(words, scores, color=colors); ax.invert_yaxis()
        ax.set_title(f'Topic {topic_id}: "{topic_name}"', fontproperties=font_prop, fontsize=16)
        ax.set_xlabel("c-TF-IDF Score", fontproperties=font_prop, fontsize=12)
        for label in ax.get_yticklabels():
            label.set_fontproperties(font_prop); label.set_fontsize(12)
        sns.despine(ax=ax); plt.tight_layout()
        safe_topic_name = "".join([c for c in topic_name if c.isalnum() or c in (' ', '_')]).rstrip()
        filename = f"{platform}_barchart_topic_{topic_id}_{safe_topic_name[:20]}.png"
        plt.savefig(os.path.join(VIS_OUTPUT_DIR, filename), bbox_inches='tight')
        plt.close(fig)
    print(f"      - 已为 {top_n_topics_to_plot} 个主题分别生成条形图。")


def generate_distance_map_plot(model, platform):
    """生成出版级质量的主题间距离图，采用对数尺寸缩放"""
    print(f"    > 正在生成 [主题间距离图 (对数尺寸缩放版)]...")
    topic_info_filtered = model.get_topic_info()[model.get_topic_info().Topic != -1]
    
    if len(topic_info_filtered) < 3:
        print(f"      - 主题数量过少 ({len(topic_info_filtered)}), 跳过。")
        return

    indices = topic_info_filtered.Topic.values + 1
    embeddings = model.topic_embeddings_[indices]
    reducer = umap.UMAP(n_neighbors=min(15, len(embeddings) - 1), min_dist=0.0, metric='cosine', random_state=42)
    coords = reducer.fit_transform(embeddings)
    
    topic_info_filtered['x'] = coords[:, 0]; topic_info_filtered['y'] = coords[:, 1]
    
    # 【核心修改】==============================================================
    # 使用对数缩放 (Log Scaling) 来计算圆圈大小，以获得更好的视觉区分度
    # 1. 对数变换：使用 log1p 避免 log(0) 的错误
    log_counts = np.log1p(topic_info_filtered['Count'].values)

    # 2. 重新定义缩放范围，让最小的圆圈也清晰可见
    min_size, max_size = 100, 1500

    # 3. 在对数变换后的数据上，再进行线性缩放
    if log_counts.max() == log_counts.min():
        # 如果所有主题大小都一样，则赋予一个中等大小
        size_scaler = np.full_like(log_counts, (min_size + max_size) / 2)
    else:
        size_scaler = (log_counts - log_counts.min()) / (log_counts.max() - log_counts.min()) * (max_size - min_size) + min_size
    # ==========================================================================

    fig, ax = plt.subplots(figsize=(16, 16))
    
    ax.scatter(
        topic_info_filtered['x'], topic_info_filtered['y'], s=size_scaler, 
        c=UNIFIED_COLOR, alpha=0.5, edgecolor="black", linewidth=0.5
    )

    texts = [ax.text(row['x'], row['y'], row['Name'], fontproperties=font_prop, fontsize=10)
             for _, row in topic_info_filtered.iterrows()]
    
    adjust_text(texts, ax=ax, expand_points=(1.5, 1.5),
                arrowprops=dict(arrowstyle="-", color='gray', lw=0.5, alpha=0.8))

    ax.set_xlabel("UMAP Dimension 1", fontproperties=font_prop, fontsize=14)
    ax.set_ylabel("UMAP Dimension 2", fontproperties=font_prop, fontsize=14)
    ax.set_title(f"{platform.capitalize()} - Inter-Topic Distance Map (UMAP Projection)", 
                 fontproperties=font_prop, fontsize=20, pad=20)
    
    ax.grid(True, linestyle='--', alpha=0.3)
    sns.despine(ax=ax)
    
    plt.savefig(os.path.join(VIS_OUTPUT_DIR, f"{platform}_intertopic_map_publication.png"), bbox_inches='tight')
    plt.savefig(os.path.join(VIS_OUTPUT_DIR, f"{platform}_intertopic_map_publication.pdf"), bbox_inches='tight')
    plt.close()


def generate_hierarchy_plot(model, platform):
    print(f"    > 正在生成 [主题层次聚类图]...")
    # ... (此函数无变化)
    try:
        if not hasattr(model, 'linkage_matrix_'):
            print("      - [错误] 模型中未找到linkage_matrix_。")
            return
        linkage_matrix = model.linkage_matrix_
        valid_topics = sorted([t for t in model.get_topics().keys() if t != -1])
        if not valid_topics: return
        labels = [model.get_topic_info(t)['Name'].iloc[0] for t in valid_topics]
        plt.figure(figsize=(10, max(8, len(labels) * 0.5)))
        with plt.rc_context({'font.family': font_prop.get_family(), 'font.sans-serif': [font_prop.get_name()]}):
             sch.dendrogram(linkage_matrix, orientation="left", labels=labels, leaf_font_size=10, color_threshold=1)
        plt.title(f"{platform.capitalize()} - Hierarchical Topic Structure", fontproperties=font_prop, fontsize=18)
        plt.xlabel("Distance", fontproperties=font_prop, fontsize=14)
        plt.grid(axis='x', linestyle='--', alpha=0.6); sns.despine(left=True, bottom=True)
        plt.tight_layout()
        plt.savefig(os.path.join(VIS_OUTPUT_DIR, f"{platform}_hierarchy.png"), bbox_inches='tight')
        plt.savefig(os.path.join(VIS_OUTPUT_DIR, f"{platform}_hierarchy.pdf"), bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"      - [错误] 生成层次聚类图失败: {e}")

def main():
    print("\n--- 启动【脚本 2: 独立可视化 (最终版 v2)】---")
    os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)
    
    print(f"\n> 正在从 '{ORIGINAL_DATA_PATH}' 加载原始文档...")
    try:
        df_original = pd.read_excel(ORIGINAL_DATA_PATH)
    except FileNotFoundError:
        print(f"  > [严重错误] 原始数据文件未找到: '{ORIGINAL_DATA_PATH}'。程序终止。")
        return

    print(f"\n> 正在预加载句子模型: '{EMBEDDING_MODEL_NAME}'...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    visualize_single_platform('bilibili', df_original, embedding_model)
    visualize_single_platform('youtube', df_original, embedding_model)

    print("\n--- 脚本 2 执行完毕！所有可视化图表已生成。---")
    print(f"图表已保存至: '{VIS_OUTPUT_DIR}'")

if __name__ == "__main__":
    main()