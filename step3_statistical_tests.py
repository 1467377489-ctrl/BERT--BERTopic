# 文件名: step3_statistical_tests.py (已修正)

import pandas as pd
from scipy.stats import mannwhitneyu
import os

# --- 配置区 ---
# 【核心修正】确保输入文件名与 step2 的输出文件名一致
INPUT_FILE_PATH = 'results/final_data_with_aspects.xlsx'
OUTPUT_REPORT_PATH = 'results/statistical_report.txt'
PLATFORM_COLUMN = 'platform'

def main():
    print("\n--- 启动【步骤3】: 统计显著性检验 ---")
    
    print("  > 正在加载最终分析数据...")
    try:
        df = pd.read_excel(INPUT_FILE_PATH)
    except FileNotFoundError:
        print(f"错误：找不到输入文件 '{INPUT_FILE_PATH}'。")
        print("请确保已成功运行 step1 和 step2 脚本。")
        return
    
    report_content = []
    
    print("  > 正在执行统计检验...")
    df_bili = df[df[PLATFORM_COLUMN].str.contains('bilibili', case=False, na=False)]
    df_yt = df[df[PLATFORM_COLUMN].str.contains('YouTube', case=False, na=False)]
    
    if df_bili.empty or df_yt.empty:
        print("错误：数据中未能同时找到 Bilibili 和 YouTube 的评论，无法进行统计检验。")
        return
        
    # 总体检验
    report_content.append("="*70)
    report_content.append("                  统计检验与对比分析报告")
    report_content.append("="*70)
    report_content.append("\n--- 1. 总体情感分数分布检验 (Mann-Whitney U Test) ---")
    u_stat, p_val = mannwhitneyu(df_bili['sentiment_score'], df_yt['sentiment_score'], alternative='two-sided')
    report_content.append(f"  - Bilibili 平均分: {df_bili['sentiment_score'].mean():.3f}")
    report_content.append(f"  - YouTube  平均分: {df_yt['sentiment_score'].mean():.3f}")
    report_content.append(f"  - P-value: {p_val:.4f}")
    report_content.append("  - 结论: " + ("两个平台总体情感分数存在【统计学显著差异】。" if p_val < 0.05 else "两个平台总体情感分数【无】统计学显著差异。"))
    
    # 方面检验
    report_content.append("\n--- 2. 各方面情感分数对比与检验 ---")
    all_aspects = sorted(df['aspect'].unique())
    for aspect in all_aspects:
        report_content.append(f"\n  【方面: {aspect}】")
        aspect_bili = df_bili[df_bili['aspect'] == aspect]['sentiment_score']
        aspect_yt = df_yt[df_yt['aspect'] == aspect]['sentiment_score']
        
        if len(aspect_bili) > 10 and len(aspect_yt) > 10:
            report_content.append(f"    - Bilibili: 平均分 {aspect_bili.mean():.3f} (N={len(aspect_bili)})")
            report_content.append(f"    - YouTube:  平均分 {aspect_yt.mean():.3f} (N={len(aspect_yt)})")
            u_stat_aspect, p_val_aspect = mannwhitneyu(aspect_bili, aspect_yt)
            report_content.append(f"    - P-value: {p_val_aspect:.4f}")
            report_content.append("    - 结论: " + ("在此方面存在【统计学显著差异】。" if p_val_aspect < 0.05 else "在此方面【无】统计学显著差异。"))
        else:
            report_content.append("    - 数据量不足，跳过统计检验。")
    report_content.append("\n" + "="*70)

    print(f"  > 正在将报告写入文件: {OUTPUT_REPORT_PATH}")
    with open(OUTPUT_REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_content))
        
    print("\n--- 步骤3完成 ---")
    print("统计报告已生成。整个分析流程结束！")

if __name__ == "__main__":
    main()