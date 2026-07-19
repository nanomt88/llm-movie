import pandas as pd
import os

# 读取CSV文件
csv_path = '/data/conv/data_all.csv'
df = pd.read_csv(csv_path)

# 过滤出用户数据 (is_seeker=True)
seeker_data = df[df['is_seeker'] == True]

# 获取所有唯一的user_id，并过滤掉空值
unique_user_ids = seeker_data['user_id'].dropna().unique()

# 输出统计信息
print(f"\n总记录数: {len(df)}")
print(f"用户记录数 (is_seeker=True): {len(seeker_data)}")
print(f"唯一用户数: {len(unique_user_ids)}")

# 确保输出目录存在
output_dir = '/data/conv'
os.makedirs(output_dir, exist_ok=True)

# 将用户ID保存到文件
output_file = os.path.join(output_dir, 'all_user_id_list.txt')
with open(output_file, 'w', encoding='utf-8') as f:
    for user_id in sorted(unique_user_ids, key = lambda x: str(x)):
        f.write(f"{user_id}\n")

print(f"\n用户ID已保存到: {output_file}")
