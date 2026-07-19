# -*- coding: utf-8 -*-
"""
使用 pandas 读取 test-min-data.csv，将每一行解析为结构化的 MovieRecord 对象。
"""

import ast
from typing import Optional

import pandas as pd




def load_records_pd(csv_path: str):
    """
    读取 CSV 文件，返回 MovieRecord 对象列表。
    """
    df = pd.read_csv(csv_path)
    return df

def print_data_frame(df: pd.DataFrame):
    print("\n=== 打印表头 ===")
    print(df.head())

    # 方法 2: 打印特定行的数据
    print("\n=== 第一行数据 ===")
    print(df.iloc[0])
    print("\n=== 第一行的 processed ===")
    print(df.iloc[0]['processed'])

    # 方法 5: 查看 DataFrame 的基本信息
    print("\n=== DataFrame 信息 ===")
    print(f"形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    print(f"\n数据类型:\n{df.dtypes}")

def extrace_user_speak(processed_text: str):
    parsed = ast.literal_eval(processed_text)
    if isinstance(parsed, list) and len(parsed) >= 2:
        role = parsed[0]  # 'USER' 或 'SYSTEM'
        content = parsed[1]  # 实际说话内容
        #print(f"{i}: 角色={role}")
        #print(f"   内容={content[:100]}...")  # 只显示前100个字符
        return content
    else:
        print(f"{i}: 解析失败 - 格式不正确: {parsed}")
        return  None
# ------------------------------------------------------------------
# 使用示例
# ------------------------------------------------------------------
if __name__ == '__main__':
    import os

    base = os.path.dirname(__file__)
    csv_path = os.path.join(base, 'data', 'test-min-data.csv')

    df = load_records_pd(csv_path)

    # 求助者（用户）的发言
    user_df = df[df['is_seeker'] == True]
    # 推荐者（系统）的发言
    system_df = df[df['is_seeker'] == False]

    #print_data_frame(user_df)

    processed_array = user_df['processed'].values

    for i in range(min(10, len(processed_array))):
        processed = processed_array[i]
        print(extrace_user_speak(processed))

    print(processed_array.size)