# 文件名: step2b_generate_plot.py
# 描述: 读取带有方面分析的数据，生成出版级质量的全英文可视化图表。

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- 配置区 ---
# 输入文件是上一步的输出
INPUT_FILE_PATH = 'results/final_data_with_aspects.xlsx' 
OUTPUT_CHART_PATH = 'results/aspect_sentiment_boxplot_english.png'

# 绘图所需的列名
PLATFORM_COLUMN = 'platform'

def create_publication_quality_boxplot(df):
    """
    根据给定的DataFrame，生成一张符合学术出版标准的全英文分组箱线图。
    """
    # --- 中文方面到英文图例的映射字典 ---
    ASPECT_NAME_MAPPING = {
        "表演-技巧呈现": "Performance-Technique",
        "表演-团队协作": "Performance-Teamwork",
        "表演-艺术表现力": "Performance-Aesthetics",
        "表演-听觉体验": "Performance-Auditory",
        "阐释-文化价值": "Interpretation-Cultural Value",
        "阐释-情感投射": "Interpretation-Emotion",
        "关联-传播与发展": "Context-Spread & Development",
        "关联-比较与记忆": "Context-Comparison & Memory",
        "综合评价与互动": "Overall Evaluation",
        "综合讨论": "General Discussion"
    }
    
    # 1. 数据准备
    # 筛选出评论数大于20的方面，以确保箱线图具有统计意义
    aspect_counts = df['aspect'].value_counts()
    valid_aspects = aspect_counts[aspect_counts > 20].index
    df_plot = df[df['aspect'].isin(valid_aspects)].copy()
    
    # 应用英文名称映射，创建一个新列用于绘图
    df_plot['aspect_en'] = df_plot['aspect'].map(ASPECT_NAME_MAPPING)
    
    # 定义绘图时X轴上各个方面的显示顺序
    aspect_order_en = [
        "Performance-Technique", "Performance-Teamwork", "Performance-Aesthetics", "Performance-Auditory",
        "Interpretation-Cultural Value", "Interpretation-Emotion",
        "Context-Spread & Development", "Context-Comparison & Memory",
        "Overall Evaluation",
        "General Discussion"
    ]
    # 仅保留数据中实际存在的方面，并维持预设顺序
    final_order = [asp for asp in aspect_order_en if asp in df_plot['aspect_en'].unique()]

    # 2. 设置Seaborn风格和调色板
    sns.set_theme(style="ticks", context="paper")
    palette = sns.color_palette("Paired", len(df_plot[PLATFORM_COLUMN].unique())) 

    # 3. 创建图表和坐标轴
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.boxplot(
        x='aspect_en',          # X轴使用英文方面名称
        y='sentiment_score',    # Y轴是情感分数
        hue=PLATFORM_COLUMN,    # 按平台进行分组
        data=df_plot, 
        order=final_order,      # 指定X轴顺序
        palette=palette,
        linewidth=1.2,          # 箱体边框线宽
        fliersize=2,            # 异常值点的大小
        ax=ax
    )
    
    # 4. 精细化调整图表元素 (全英文)
    fig.suptitle('Cross-Platform Sentiment Score Comparison by Detailed Aspect', fontsize=18, fontweight='bold', y=0.99)
    ax.set_title('Sentiment scores range from -1 (Negative) to +1 (Positive)', fontsize=11, style='italic', y=1.02)
    
    ax.set_ylabel('Sentiment Score', fontsize=13)
    ax.set_xlabel('Comment Aspect', fontsize=13)
    ax.set_ylim(-1.1, 1.1) # 设置Y轴范围
    
    # 旋转X轴标签以防重叠
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # 添加辅助网格线和零点线
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.8)

    # 自定义图例
    ax.legend(
        title='Platform', 
        fontsize=11, 
        title_fontsize=12,
        frameon=False, # 不显示图例边框
        bbox_to_anchor=(1.02, 1),
        loc='upper left'
    )
    
    # 移除顶部和右侧的坐标轴线
    sns.despine(trim=True)
    
    # 5. 调整布局并保存图像
    plt.tight_layout(rect=[0, 0, 0.9, 0.96]) # 调整布局，为标题和图例留出空间
    plt.savefig(OUTPUT_CHART_PATH, dpi=300, bbox_inches='tight')
    print(f"  > 全英文出版级图表已保存至: '{OUTPUT_CHART_PATH}'")

def main():
    """
    主函数，执行数据加载和图表生成。
    """
    print("\n--- 启动【步骤2b】: 生成方面情感可视化图表 ---")
    
    # 检查输入文件是否存在
    if not os.path.exists(INPUT_FILE_PATH):
        print(f"  > 错误: 输入文件 '{INPUT_FILE_PATH}' 未找到。请先运行 step2a_aspect_analysis.py。")
        return

    # 读取包含方面信息的数据
    df = pd.read_excel(INPUT_FILE_PATH)
    print(f"  > 已加载 {len(df)} 条带方面标签的评论。")
    
    # 调用函数生成图表
    print("  > 正在生成出版级质量的全英文箱线图...")
    create_publication_quality_boxplot(df)
    
    print(f"\n--- 步骤2b完成 ---")
    print("  > 下一步，请运行 step3_statistical_tests.py 进行统计检验。")

if __name__ == "__main__":
    main()