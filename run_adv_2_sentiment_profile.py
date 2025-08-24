# 文件名: run_adv_2_batch_sentiment_profiles.py
# 功能: 自动为数据中所有方面批量生成情感剖面密度图。

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# --- 配置区 ---
INPUT_FILE = 'results/final_data_with_aspects.xlsx'
# 【核心修改】输出目录将按方面自动创建子文件夹
BASE_OUTPUT_DIR = 'results/advanced_analysis/sentiment_profiles_by_aspect/'
PLATFORM_COLUMN = 'platform'
ASPECT_COLUMN = 'aspect'
SENTIMENT_COLUMN = 'sentiment_score'

# 翻译映射字典 (保持不变)
ASPECT_TRANSLATION_MAP = {
    "表演-技巧呈现": "Performance-Technique",
    "表演-团队协作": "Performance-Teamwork",
    "表演-艺术表现力": "Performance-Aesthetics",
    "表演-听觉体验": "Performance-Auditory",
    "阐释-文化价值": "Interpretation-Cultural Value",
    "阐释-情感投射": "Interpretation-Emotion",
    "关联-传播与发展": "Context-Spread & Development",
    "关联-比较与联想": "Context-Comparison & Lenovo", # 已修正Lenovo笔误
    "综合讨论": "General Discussion"
}

# --- 主函数 (核心修改) ---
def main():
    """主函数，循环遍历所有方面并生成图表"""
    print("--- 启动进阶分析2: 全方面情感剖面批量生成 ---")
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    # 1. 加载数据
    try:
        df = pd.read_excel(INPUT_FILE)
    except FileNotFoundError:
        print(f"错误：找不到输入文件 '{INPUT_FILE}'。")
        return

    # 2. 【核心修改】获取所有唯一的方面，并进行循环
    # .unique() 会返回该列所有不重复的值
    all_aspects_chinese = df[ASPECT_COLUMN].unique()
    
    print(f"  > 将为以下 {len(all_aspects_chinese)} 个方面生成图表: {list(all_aspects_chinese)}")
    
    for aspect_chinese in all_aspects_chinese:
        # 排除评论数过少的方面，避免生成无意义的图表
        if len(df[df[ASPECT_COLUMN] == aspect_chinese]) < 50: # 一个方面至少需要50条评论
            print(f"\n--- 跳过方面 ‘{aspect_chinese}’ (数据量不足) ---")
            continue

        aspect_english = ASPECT_TRANSLATION_MAP.get(aspect_chinese, aspect_chinese)
        print(f"\n--- 正在处理方面: ‘{aspect_english}’ ---")
        
        # 筛选特定方面的数据
        df_aspect = df[df[ASPECT_COLUMN] == aspect_chinese]
        
        # 调用绘图函数
        create_density_plot(df_aspect, aspect_english)
        
    print("\n--- 所有图表生成完成！ ---")
    print(f"所有情感剖面图已保存至: '{BASE_OUTPUT_DIR}' 文件夹。")
    
def create_density_plot(data, aspect_name_english):
    """生成一个美化的、带清晰图例的全英文密度图"""
    plt.rcParams['font.family'] = 'Arial'
    sns.set_theme(style="whitegrid", context="paper")

    platform_colors = {
        "YouTube": "#F8766D",
        "bilibili": "#619CFF"
    }

    plt.figure(figsize=(10, 6))
    
    ax = sns.kdeplot(
        data=data, 
        x=SENTIMENT_COLUMN, 
        hue=PLATFORM_COLUMN, 
        fill=True, 
        common_norm=False,
        palette=platform_colors,
        alpha=0.5,
        linewidth=2
    )
    
    # 手动创建和放置图例
    legend_patches = [mpatches.Patch(color=color, label=platform) 
                      for platform, color in platform_colors.items() 
                      if platform in data[PLATFORM_COLUMN].unique()] # 只为数据中存在的平台创建图例

    plt.legend(handles=legend_patches, 
               title='Platform', 
               frameon=False, 
               loc='upper left', 
               bbox_to_anchor=(1.01, 1))

    # 图表美化
    ax.set_title(f'Sentiment Score Density Profile for Aspect: "{aspect_name_english}"', fontsize=16, fontweight='bold')
    ax.set_xlabel('Sentiment Score (-1: Negative to +1: Positive)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.axvline(0, color='black', linestyle='--', alpha=0.7, linewidth=1)
    sns.despine()
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    # 保存文件
    safe_filename = aspect_name_english.replace(" & ", "_and_").replace("-", "_").lower()
    output_filename = f'sentiment_profile_{safe_filename}.png'
    plt.savefig(os.path.join(BASE_OUTPUT_DIR, output_filename), dpi=300, bbox_inches='tight')
    plt.close() # 关闭图形，释放内存，对于循环绘图至关重要

if __name__ == "__main__":
    main()