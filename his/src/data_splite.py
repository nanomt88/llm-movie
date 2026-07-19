import pandas as pd
import os
import numpy as np

def sample_adjacent_pairs(csv_path, sample_ratio=0.05):
    """
    将 CSV 视为成对的行（如 Row 0-1 为一组，Row 2-3 为一组），
    随机提取指定比例的组。
    """
    print(f"正在加载数据: {csv_path}")
    df = pd.read_csv(csv_path)
    total_rows = len(df)

    # 确保行数是偶数，如果是奇数则丢弃最后一行
    if total_rows % 2 != 0:
        df = df.iloc[:-1]
        total_rows = len(df)

    num_pairs = total_rows // 2
    sample_pairs_count = max(1, int(num_pairs * sample_ratio))

    print(f"总对话对数: {num_pairs}, 计划提取: {sample_pairs_count} 对")

    # 随机选择要保留的“对”的索引
    selected_pair_indices = np.random.choice(num_pairs, size=sample_pairs_count, replace=False)
    selected_pair_indices.sort()

    # 将“对”的索引转换为实际的行索引
    # 例如第 0 对对应行索引 [0, 1]，第 1 对对应 [2, 3]
    rows_to_keep = []
    for pair_idx in selected_pair_indices:
        rows_to_keep.append(pair_idx * 2)     # 第一行
        rows_to_keep.append(pair_idx * 2 + 1) # 第二行

    # 提取这些行
    sampled_df = df.iloc[rows_to_keep].reset_index(drop=True)

    # 保存结果
    output_filename = f"my-train-{int(sample_ratio*100)}percent-data.csv"
    output_path = os.path.join(os.path.dirname(csv_path), output_filename)
    sampled_df.to_csv(output_path, index=False)

    print(f"已提取 {len(sampled_df)} 行数据")
    print(f"文件已保存至: {output_path}")

# 使用示例
if __name__ == '__main__':
    file_path = r"/data/my-tran-holidy-data.csv"
    sample_adjacent_pairs(file_path, sample_ratio=0.2)
