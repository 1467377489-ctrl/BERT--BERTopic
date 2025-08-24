# 文件名: run_adv_3_sub_topic_modeling.py

import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# --- 配置区 ---
INPUT_FILE = 'results/final_integrated_dataset.xlsx'
OUTPUT_DIR = 'results/advanced_analysis/sub_topics/'
PLATFORM_COLUMN = 'platform'
ASPECT_COLUMN = 'aspect'
COMMENT_COLUMN = 'cleaned_text_for_sentiment'

def main():
    """主函数，对'综合讨论'类别进行二次主题建模"""
    print("--- 启动进阶分析3: ‘综合讨论’的二次主题建模 ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 加载数据并筛选
    try:
        df = pd.read_excel(INPUT_FILE)
    except FileNotFoundError:
        print(f"错误：找不到输入文件 '{INPUT_FILE}'。")
        return
        
    df_general = df[df[ASPECT_COLUMN] == '综合讨论']
    print(f"  > 已筛选出 {len(df_general)} 条‘综合讨论’评论进行分析。")

    # 2. 分别为两个平台进行子主题建模
    for platform in df_general[PLATFORM_COLUMN].unique():
        print(f"\n--- 正在为平台‘{platform}’的‘综合讨论’建模 ---")
        docs_platform_general = df_general[df_general[PLATFORM_COLUMN] == platform][COMMENT_COLUMN].tolist()
        
        if len(docs_platform_general) < 50:
            print(f"  > 平台‘{platform}’的‘综合讨论’评论数不足，跳过。")
            continue
            
        # 借用 step4 的建模和可视化函数
        from step4_topic_modeling import run_bertopic_for_platform, save_topic_info
        sub_topic_model = run_bertopic_for_platform(docs_platform_general)
        
        # 保存结果到新的子目录
        platform_output_dir = os.path.join(OUTPUT_DIR, platform)
        os.makedirs(platform_output_dir, exist_ok=True)
        topic_info_df = sub_topic_model.get_topic_info()
        topic_info_df.to_excel(os.path.join(platform_output_dir, 'sub_topics_info.xlsx'), index=False)
        print(f"  > 平台‘{platform}’的子主题信息表已保存。")
        
        # 这里可以添加一个简单的可视化，或直接分析Excel结果
        
    print("\n--- 分析完成 ---")
    print(f"所有子主题建模结果已保存至: '{OUTPUT_DIR}' 文件夹。")

if __name__ == "__main__":
    # 重要提示：这个脚本依赖于 step4_topic_modeling.py 中的函数。
    # 请确保这两个文件在同一个目录下。
    try:
        import step4_topic_modeling
    except ImportError:
        print("错误：无法导入 'step4_topic_modeling.py'。请确保该文件存在于当前目录。")
        exit()
    main()