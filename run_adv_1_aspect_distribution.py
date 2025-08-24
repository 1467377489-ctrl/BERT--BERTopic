# 文件名: run_adv_1_aspect_distribution_v2.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- 配置区 ---
INPUT_FILE = 'results/final_data_with_aspects.xlsx'
OUTPUT_DIR = 'results/advanced_analysis/'
PLATFORM_COLUMN = 'platform'
ASPECT_COLUMN = 'aspect'

# 中文到英文的方面标签映射字典
ASPECT_TRANSLATION_MAP = {
    "表演-技巧呈现": "Performance-Technique",
    "表演-团队协作": "Performance-Teamwork",
    "表演-艺术表现力": "Performance-Aesthetics",
    "表演-听觉体验": "Performance-Auditory",
    "阐释-文化价值": "Interpretation-Cultural Value",
    "阐释-情感投射": "Interpretation-Emotion",
    "关联-传播与发展": "Context-Spread & Development",
    "关联-比较与记忆": "Context-Comparison & Memory",
    "综合讨论": "General Discussion"
}

def main():
    """主函数，执行方面分布分析并生成两张图"""
    print("--- 启动进阶分析1 v2.0: 拆分叙事的方面分布对比 ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    df = pd.read_excel(INPUT_FILE)
    df['aspect_english'] = df[ASPECT_COLUMN].map(ASPECT_TRANSLATION_MAP).fillna('Other')
    
    # --- 为图一：宏观结构图准备数据 ---
    print("  > 正在准备宏观结构图的数据...")
    df_overview = df.copy()
    specific_aspects = [v for k, v in ASPECT_TRANSLATION_MAP.items() if k != "综合讨论"]
    df_overview['structure_category'] = df_overview['aspect_english'].apply(
        lambda x: 'Specific Thematic Discussion' if x in specific_aspects else x
    )
    overview_counts = df_overview.groupby([PLATFORM_COLUMN, 'structure_category']).size().reset_index(name='count')
    total_overview = overview_counts.groupby(PLATFORM_COLUMN)['count'].transform('sum')
    overview_counts['percentage'] = (overview_counts['count'] / total_overview) * 100

    # --- 为图二：具体方面深度对比图准备数据 ---
    print("  > 正在准备具体方面深度对比图的数据...")
    df_specific = df[df['aspect_english'].isin(specific_aspects)]
    specific_counts = df_specific.groupby([PLATFORM_COLUMN, 'aspect_english']).size().reset_index(name='count')
    total_specific = specific_counts.groupby(PLATFORM_COLUMN)['count'].transform('sum')
    specific_counts['percentage'] = (specific_counts['count'] / total_specific) * 100

    # --- 生成并组合图表 ---
    print("  > 正在生成组合图表...")
    create_combined_plot(overview_counts, specific_counts)
    
    print("\n--- 分析完成 ---")
    print(f"组合图表已保存至: '{os.path.join(OUTPUT_DIR, 'aspect_distribution_combined_plot.png')}'")

def create_combined_plot(overview_data, specific_data):
    """将两张图绘制在一张大的Figure上"""
    plt.rcParams['font.family'] = 'Arial'
    sns.set_theme(style="ticks", context="paper")

    fig, axes = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [1, 2]})
    
    # --- 绘制图一：宏观结构图 ---
    sns.barplot(data=overview_data, x='percentage', y='structure_category', hue=PLATFORM_COLUMN,
                palette="viridis", orient='h', ax=axes[0])
    axes[0].set_title('A: Overall Structure of Discussions', loc='left', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Percentage of All Comments (%)', fontsize=11)
    axes[0].set_ylabel('')
    axes[0].legend(title='Platform', frameon=False)
    axes[0].set_xlim(0, 100)
    for p in axes[0].patches:
        axes[0].text(p.get_width() - 1, p.get_y() + p.get_height() / 2, f'{p.get_width():.1f}%', 
                     ha='right', va='center', color='white', fontweight='bold')

    # --- 绘制图二：具体方面深度对比图 ---
    order = specific_data.groupby('aspect_english')['percentage'].sum().sort_values(ascending=False).index
    sns.barplot(data=specific_data, x='percentage', y='aspect_english', hue=PLATFORM_COLUMN,
                palette="viridis", orient='h', order=order, ax=axes[1])
    axes[1].set_title('B: Distribution within Specific Thematic Discussions', loc='left', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Percentage of Thematic Comments (%)', fontsize=11)
    axes[1].set_ylabel('')
    axes[1].get_legend().remove() # 移除第二个图的图例，因为和第一个一样
    
    # 清理视觉元素
    sns.despine(fig=fig)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'aspect_distribution_combined_plot.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    main()