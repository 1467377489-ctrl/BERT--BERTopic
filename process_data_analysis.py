
# 文件名: preprocess_data_enhanced.py

import pandas as pd
import re
import emoji
from opencc import OpenCC
import os

# --- 配置区 ---
# 原始数据文件路径
RAW_DATA_PATH = 'data/final_data.xlsx'
# 清洗后数据保存路径
CLEANED_DATA_PATH = 'data/comments_for_sentiment.xlsx'
# 评论所在的列名
COMMENT_COLUMN_NAME = 'comment_text'


# --- 初始化全局组件 ---
# 在脚本的全局区域，只初始化一次OpenCC转换器以提高效率。
# 同时增加了严格的错误处理机制。
print("正在初始化简繁转换器...")
try:
    # 't2s' 表示 Traditional Chinese to Simplified Chinese
    cc = OpenCC('t2s') 
    print("简繁转换器初始化成功。")
except Exception as e:
    print(f"致命错误：无法初始化OpenCC简繁转换器: {e}")
    print("请确保 'opencc-python-reimplemented' 库已正确安装 (pip install opencc-python-reimplemented)。")
    print("程序将终止，请解决此问题后再运行。")
    exit() # 直接退出程序，防止后续错误


# --- 核心清洗函数 (为情感分析深度定制) ---
def clean_for_sentiment_analysis(text):
    """
    对单条评论文本进行清洗，专门优化社交媒体和视频评论的情感分析任务。
    
    清洗步骤:
    1.  处理非字符串输入。
    2.  繁体转简体。
    3.  移除URL链接。
    4.  移除@提及。
    5.  移除#话题#。
    6.  移除视频时间戳 (如 03:15, 1:20:35)。
    7.  缩减过度重复的字符 (如 "哇!!!!" -> "哇!!")。
    8.  将表情符号(emoji)转为文字描述 (保留情感信息)。
    9.  规范化中英文标点。
    10. 移除多余的空格和首尾空格。
    """
    # 1. 确保输入为字符串，处理空值(NaN)等情况
    if not isinstance(text, str):
        return ""
        
    # 2. 繁体转简体
    text = cc.convert(text)
    
    # 3. 移除URL
    text = re.sub(r'http\S+|https\S+', '', text)
    
    # 4. 移除@提及
    text = re.sub(r'@\S+', '', text)
    
    # 5. 移除#话题#
    text = re.sub(r'#\S+#', '', text)
    
    # 6. 【增强】移除视频时间戳
    text = re.sub(r'\d{1,2}:\d{1,2}(:\d{1,2})?', '', text)
    
    # 7. 【增强】将3次以上的连续重复字符缩减为2次，以保留情感强度
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # 8. 将emoji转换为文字描述，例如 "😂" -> ":face_with_tears_of_joy:"
    text = emoji.demojize(text)
    
    # 9. 规范化常用标点 (全角转半角)
    text = text.replace('！', '!').replace('？', '?').replace('，', ',').replace('。', '.')
    
    # 10. 将多个连续空格替换为单个空格，并移除首尾空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# --- 主流程 ---
def main():
    """
    执行数据清洗的主流程：读取 -> 清洗 -> 保存
    """
    print("\n--- 启动为【情感分析】定制的数据清洗脚本 ---")
    
    # 检查原始数据文件是否存在
    if not os.path.exists(RAW_DATA_PATH):
        print(f"错误：找不到原始数据文件 '{RAW_DATA_PATH}'。")
        print("请确保原始数据文件已放置在正确的位置。")
        return

    print(f"\n[步骤 1/4] 正在从 '{RAW_DATA_PATH}' 读取原始数据...")
    try:
        df = pd.read_excel(RAW_DATA_PATH)
        print(f"读取成功，原始数据共 {len(df)} 行。")
    except Exception as e:
        print(f"读取Excel文件时发生错误: {e}")
        return

    # 检查评论列是否存在
    if COMMENT_COLUMN_NAME not in df.columns:
        print(f"错误：在Excel文件中找不到名为 '{COMMENT_COLUMN_NAME}' 的列。")
        print(f"可用的列有: {list(df.columns)}")
        return

    print(f"\n[步骤 2/4] 正在对 '{COMMENT_COLUMN_NAME}' 列的文本进行清洗...")
    # 使用 .apply() 方法将清洗函数应用到每一行评论
    df['cleaned_text'] = df[COMMENT_COLUMN_NAME].apply(clean_for_sentiment_analysis)
    print("文本清洗完成。")
    
    print("\n[步骤 3/4] 正在移除清洗后为空的评论...")
    original_rows = len(df)
    
    # 移除内容为空的行，可以链式操作更简洁
    df.dropna(subset=['cleaned_text'], inplace=True)
    df = df[df['cleaned_text'].str.strip() != '']
    
    cleaned_rows = len(df)
    removed_count = original_rows - cleaned_rows
    if removed_count > 0:
        print(f"移除了 {removed_count} 条无效或空的评论。剩余 {cleaned_rows} 条有效评论。")
    else:
        print("所有评论均为有效评论，未移除任何行。")

    print(f"\n[步骤 4/4] 正在将清洗后的数据保存至 '{CLEANED_DATA_PATH}'...")
    # 确保输出目录存在
    output_dir = os.path.dirname(CLEANED_DATA_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 保存到Excel
    df.to_excel(CLEANED_DATA_PATH, index=False, engine='openpyxl')
    
    print("\n--- 清洗流程成功结束 ---")
    print(f"结果已保存至: '{CLEANED_DATA_PATH}'")

# --- 脚本执行入口 ---
if __name__ == "__main__":
    main()