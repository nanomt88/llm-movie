import sys
import io

# 设置 stdout 编码为 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('data/yearly/all_holiday_records_v3.csv')

# 提取会话 ID（conv_id 前9个字符）
df['session_id'] = df['conv_id'].str[:9]

# 1. 会话总数
total_sessions = df['session_id'].nunique()
print(f"=" * 60)
print(f"{'会话数据分析报告':^58}")
print(f"=" * 60)
print(f"\n1. 总会话数量: {total_sessions}\n")

# 2. 按会话分组统计
session_stats = []

for session_id, group in df.groupby('session_id'):
    total_rows = len(group)
    user_q_count = group[group['is_seeker'] == True].shape[0]
    sys_r_count = group[group['is_seeker'] == False].shape[0]
    unique_user_q = group[group['is_seeker'] == True]['processed'].nunique()
    user_ratio = user_q_count / total_rows * 100 if total_rows > 0 else 0
    sys_ratio = sys_r_count / total_rows * 100 if total_rows > 0 else 0

    session_stats.append({
        'session_id': session_id,
        'total_rows': total_rows,
        'user_questions': user_q_count,
        'system_replies': sys_r_count,
        'unique_user_questions': unique_user_q,
        'user_ratio_pct': round(user_ratio, 2),
        'system_ratio_pct': round(sys_ratio, 2),
    })

stats_df = pd.DataFrame(session_stats)
stats_df.to_csv('data/yearly/session_info.csv',index=False, encoding='utf-8-sig')
# 打印每个会话的详细信息
print(f"{'会话ID':<14} {'总行数':>6} {'用户提问':>8} {'系统回复':>8} {'去重提问':>8} {'用户占比%':>9} {'系统占比%':>9}")
print("-" * 66)
for _, row in stats_df.iterrows():
    print(f"{row['session_id']:<14} {row['total_rows']:>6} {row['user_questions']:>8} "
          f"{row['system_replies']:>8} {row['unique_user_questions']:>8} "
          f"{row['user_ratio_pct']:>8.2f}% {row['system_ratio_pct']:>8.2f}%")

# 3. 汇总统计
print(f"\n" + "=" * 60)
print(f"{'汇总统计':^58}")
print("=" * 60)
print(f"总会话数:                              {total_sessions}")
print(f"平均每会话总行数:                      {stats_df['total_rows'].mean():.2f}")
print(f"平均每会话用户提问数:                  {stats_df['user_questions'].mean():.2f}")
print(f"平均每会话系统回复数:                  {stats_df['system_replies'].mean():.2f}")
print(f"平均每会话去重后用户提问数:            {stats_df['unique_user_questions'].mean():.2f}")
print(f"平均用户提问占比:                      {stats_df['user_ratio_pct'].mean():.2f}%")
print(f"平均系统回复占比:                      {stats_df['system_ratio_pct'].mean():.2f}%")
print("=" * 60)
