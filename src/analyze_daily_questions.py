import pandas as pd
import numpy as np
import openpyxl
from datetime import timedelta
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.holiday_util import HolidayCalendar


def analyze_daily_questions(csv_path: str = '../data/yearly/data_2021.csv'):
    """
    统计每天的用户提问数量，并进行多维度对比分析。

    返回 DataFrame，包含以下列：
        - date: 日期
        - day_of_week: 星期几
        - total_questions: 当天总提问数
        - is_holiday: 是否节假日
        - holiday_name: 节日名称
        - avg_non_holiday_month: 当月非节假日平均提问数
        - prev_week_same_day: 前一周同一天的提问数
        - prev_month_same_day: 前一个月同一天的提问数
        - ratio_vs_avg: 与月平均的比值
        - ratio_vs_prev_week: 与前一周同天的比值
        - ratio_vs_prev_month: 与前一月同天的比值
    """
    # 加载数据
    csv_path = os.path.join(os.path.dirname(__file__),csv_path)
    print(f"加载数据: {csv_path}")
    df = pd.read_csv(csv_path)

    # 解析时间戳
    df['date'] = pd.to_datetime(df['utc_time'], unit='s').dt.date

    # 筛选用户提问（is_seeker=True）
    seeker_df = df[df['is_seeker'] == True].copy()

    # 按日期分组统计
    daily_stats = seeker_df.groupby('date').size().reset_index(name='total_questions')
    daily_stats['date'] = pd.to_datetime(daily_stats['date'])
    daily_stats['day_of_week'] = daily_stats['date'].dt.day_name()

    # 加载节假日日历
    cal = HolidayCalendar()

    # 标记节假日
    daily_stats['is_holiday'] = daily_stats['date'].apply(lambda x: cal.is_holiday(x))
    daily_stats['holiday_name'] = daily_stats['date'].apply(
        lambda x: '; '.join(cal.get_holiday_names(x)) if cal.is_holiday(x) else ''
    )

    # 计算当月非节假日平均提问数
    def get_month_avg(row):
        month = row['date'].month
        year = row['date'].year

        # 当月所有日期
        month_mask = (daily_stats['date'].dt.month == month) & \
                     (daily_stats['date'].dt.year == year)

        # 当月非节假日
        non_holiday_in_month = daily_stats[month_mask & ~daily_stats['is_holiday']]

        if len(non_holiday_in_month) > 0:
            return round(non_holiday_in_month['total_questions'].mean(), 2)
        return np.nan

    daily_stats['avg_non_holiday_month'] = daily_stats.apply(get_month_avg, axis=1)

    # 计算前一周同一天的提问数
    def get_prev_week(row):
        prev_date = row['date'] - timedelta(days=7)
        prev_row = daily_stats[daily_stats['date'] == prev_date]
        if len(prev_row) > 0:
            return prev_row['total_questions'].values[0]
        return np.nan

    daily_stats['prev_week_same_day'] = daily_stats.apply(get_prev_week, axis=1)

    # 计算前一个月同一天的提问数
    def get_prev_month(row):
        current_date = row['date']
        # 处理月末日期问题
        try:
            if current_date.month == 1:
                prev_date = current_date.replace(year=current_date.year - 1, month=12)
            else:
                prev_date = current_date.replace(month=current_date.month - 1)
        except ValueError:
            # 处理如 3月31日 -> 2月28日的情况
            if current_date.month == 3 and current_date.day == 31:
                prev_date = current_date.replace(month=2, day=28)
            else:
                return np.nan

        prev_row = daily_stats[daily_stats['date'] == prev_date]
        if len(prev_row) > 0:
            return prev_row['total_questions'].values[0]
        return np.nan

    daily_stats['prev_month_same_day'] = daily_stats.apply(get_prev_month, axis=1)

    # 计算比值
    daily_stats['ratio_vs_avg'] = (daily_stats['total_questions'] /
                                   daily_stats['avg_non_holiday_month']).round(2)
    daily_stats['ratio_vs_prev_week'] = (daily_stats['total_questions'] /
                                         daily_stats['prev_week_same_day']).round(2)
    daily_stats['ratio_vs_prev_month'] = (daily_stats['total_questions'] /
                                          daily_stats['prev_month_same_day']).round(2)

    # 排序
    daily_stats = daily_stats.sort_values('date').reset_index(drop=True)

    return daily_stats


def print_analysis_report(daily_stats: pd.DataFrame):
    """打印分析报告"""
    print("\n" + "="*80)
    print("每日提问数量统计分析报告")
    print("="*80)

    # 基本信息
    print(f"\n 基本统计:")
    print(f"  总天数: {len(daily_stats)}")
    print(f"  总提问数: {daily_stats['total_questions'].sum()}")
    print(f"  日均提问数: {daily_stats['total_questions'].mean():.2f}")
    print(f"  最高单日提问: {daily_stats['total_questions'].max()} ({daily_stats.loc[daily_stats['total_questions'].idxmax(), 'date'].strftime('%Y-%m-%d')})")
    print(f"  最低单日提问: {daily_stats['total_questions'].min()} ({daily_stats.loc[daily_stats['total_questions'].idxmin(), 'date'].strftime('%Y-%m-%d')})")

    # 节假日 vs 非节假日
    print(f"\n 节假日与非节假日对比:")
    holiday_days = daily_stats[daily_stats['is_holiday'] == True]
    non_holiday_days = daily_stats[daily_stats['is_holiday'] == False]

    if len(holiday_days) > 0:
        holiday_avg = holiday_days['total_questions'].mean()
        non_holiday_avg = non_holiday_days['total_questions'].mean()
        print(f"  节假日平均提问: {holiday_avg:.2f} 条/天 ({len(holiday_days)} 天)")
        print(f"  非节假日平均提问: {non_holiday_avg:.2f} 条/天 ({len(non_holiday_days)} 天)")
        print(f"  节假日是平时的 {holiday_avg/non_holiday_avg:.2f} 倍")

    # 节假日详细列表
    if len(holiday_days) > 0:
        print(f"\n📅 节假日明细:")
        for _, row in holiday_days.iterrows():
            print(f"  {row['date'].strftime('%Y-%m-%d')} ({row['day_of_week'][:3]}): "
                  f"{row['total_questions']:4d} 条 | {row['holiday_name']}")
            if pd.notna(row['avg_non_holiday_month']):
                print(f"    ├─ 当月非节假日平均: {row['avg_non_holiday_month']:.2f} 条")
                print(f"    ├─ 是月平均的 {row['ratio_vs_avg']:.2f} 倍")
            if pd.notna(row['prev_week_same_day']):
                print(f"    ├─ 前一周同天: {int(row['prev_week_same_day'])} 条 "
                      f"(比值: {row['ratio_vs_prev_week']:.2f})")
            if pd.notna(row['prev_month_same_day']):
                print(f"    └─ 前一月同天: {int(row['prev_month_same_day'])} 条 "
                      f"(比值: {row['ratio_vs_prev_month']:.2f})")

    # Top 10 高提问日
    print(f"\n🔥 Top 10 高提问日:")
    top_10 = daily_stats.nlargest(10, 'total_questions')
    for _, row in top_10.iterrows():
        holiday_tag = f" [{row['holiday_name']}]" if row['is_holiday'] else ""
        print(f"  {row['date'].strftime('%Y-%m-%d')} ({row['day_of_week'][:3]}): "
              f"{row['total_questions']:4d} 条{holiday_tag}")

    # 按星期统计
    print(f"\n📆 按星期分布:")
    weekday_stats = daily_stats.groupby('day_of_week')['total_questions'].agg(['mean', 'sum', 'count']).round(2)
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_cn = {'Monday': '周一', 'Tuesday': '周二', 'Wednesday': '周三',
                  'Thursday': '周四', 'Friday': '周五', 'Saturday': '周六', 'Sunday': '周日'}

    for day in weekday_order:
        if day in weekday_stats.index:
            stats = weekday_stats.loc[day]
            print(f"  {weekday_cn[day]:4s}: 平均 {stats['mean']:6.2f} 条, "
                  f"总计 {int(stats['sum']):5d} 条 ({int(stats['count']):3d} 天)")

    print("\n" + "="*80)


def save_to_excel(daily_stats: pd.DataFrame, output_path: str = 'data/daily_analysis.xlsx'):
    """保存分析结果到 Excel"""
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # 完整数据
        daily_stats.to_excel(writer, sheet_name='每日统计', index=False)

        # 节假日数据
        holiday_data = daily_stats[daily_stats['is_holiday'] == True]
        if len(holiday_data) > 0:
            holiday_data.to_excel(writer, sheet_name='节假日详情', index=False)

        # Top 10
        top_10 = daily_stats.nlargest(10, 'total_questions')
        top_10.to_excel(writer, sheet_name='Top10高提问日', index=False)

        # 按星期统计
        weekday_stats = daily_stats.groupby('day_of_week')['total_questions'].agg(['mean', 'sum', 'count']).round(2)
        weekday_stats.to_excel(writer, sheet_name='星期分布')

    print(f"\n✓ 分析结果已保存到: {output_path}")


if __name__ == '__main__':
    # 分析单个文件
    csv_path = '../data/yearly/data_2021.csv'

    if os.path.exists(csv_path):
        # 执行分析
        result = analyze_daily_questions(csv_path)

        # 打印报告
        print_analysis_report(result)

        # 保存到 Excel
        save_to_excel(result, csv_path.replace('.csv', '_analysis.xlsx'))

        # 查看原始数据
        print(f"\n📋 原始数据预览 (前10行):")
        print(result[['date', 'day_of_week', 'total_questions', 'is_holiday',
                      'holiday_name', 'avg_non_holiday_month']].head(10).to_string())
    else:
        print(f"错误: 文件不存在 - {csv_path}")
