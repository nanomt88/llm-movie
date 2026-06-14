# -*- coding: utf-8 -*-
"""
节假日工具模块：加载 holiday.csv 并提供日期是否为节假日的判断功能。
可与 extrace.py 中的 MovieRecord / pd.DataFrame 配合使用。
"""

import os
from datetime import date as Date, datetime
from typing import Optional

import pandas as pd


class Holiday:
    """单条节假日记录"""

    def __init__(self, date: Date, description: str, type_: str):
        self.date = date                # 日期
        self.description = description  # 节日名称
        self.type = type_               # 节日类型（如"法定节假日""传统习俗节日"等）

    def __repr__(self) -> str:
        return f"Holiday({self.date}, {self.description})"


class HolidayCalendar:
    """
    节假日日历：加载 holiday.csv，提供日期查询接口。
    """

    def __init__(self, csv_path: Optional[str] = None):
        # 默认路径：与本文件同级的 data/holiday.csv
        if csv_path is None:
            csv_path = os.path.join(os.path.dirname(__file__), '../data', 'holiday.csv')

        # 读取 CSV
        raw = pd.read_csv(csv_path, dtype=str)

        # 解析为 Holiday 对象
        self._holidays: list[Holiday] = []
        for _, r in raw.iterrows():
            try:
                d = Date.fromisoformat(r['date'])
            except (ValueError, KeyError):
                continue
            self._holidays.append(Holiday(
                date=d,
                description=r.get('description', ''),
                type_=r.get('type', ''),
            ))

        # 构建快速查询索引：date -> list[Holiday]
        self._index: dict[Date, list[Holiday]] = {}
        for h in self._holidays:
            self._index.setdefault(h.date, []).append(h)

    # ------------------------------------------------------------------
    # 公开查询接口
    # ------------------------------------------------------------------

    @property
    def all_holidays(self) -> list[Holiday]:
        """返回所有节假日"""
        return list(self._holidays)

    @property
    def dates(self) -> set[Date]:
        """返回所有节假日日期的集合"""
        return set(self._index.keys())

    def is_holiday(self, dt) -> bool:
        """
        判断给定日期是否为节假日。

        参数:
            dt: 可以是 datetime、date、或 'YYYY-MM-DD' 格式字符串
        返回:
            True / False
        """
        d = self._to_date(dt)
        return d in self._index

    def get_holidays(self, dt) -> list[Holiday]:
        """
        获取给定日期对应的节假日信息（可能有多个）。

        参数:
            dt: 可以是 datetime、date、或 'YYYY-MM-DD' 格式字符串
        返回:
            该日期的节假日列表，非节假日返回空列表
        """
        d = self._to_date(dt)
        return list(self._index.get(d, []))

    def get_holiday_names(self, dt) -> list[str]:
        """获取给定日期的节日名称列表"""
        return [h.description for h in self.get_holidays(dt)]

    def get_holiday_types(self, dt) -> list[str]:
        """获取给定日期的节日类型列表"""
        return [h.type for h in self.get_holidays(dt)]

    def filter_by_type(self, type_: str) -> list[Holiday]:
        """按节日类型过滤（如 '法定节假日'）"""
        return [h for h in self._holidays if h.type == type_]

    def between(self, start, end) -> list[Holiday]:
        """返回日期范围内的所有节假日（含两端）"""
        s = self._to_date(start)
        e = self._to_date(end)
        return [h for h in self._holidays if s <= h.date <= e]

    # ------------------------------------------------------------------
    # 配合 MovieRecord / pd.DataFrame 使用的批量接口
    # ------------------------------------------------------------------

    def add_holiday_flags(self, df: pd.DataFrame, col: str = 'utc_time') -> pd.DataFrame:
        """
        给 DataFrame 添加节假日标记列。

        参数:
            df:  源 DataFrame（必须包含 col 列）
            col: 时间列名，可以是 utc 时间戳(int) 或 'YYYY-MM-DD' 字符串
        返回:
            新增了 holiday / holiday_name / holiday_type 三列的 DataFrame
        """
        result = df.copy()

        def _flag(row):
            # 尝试从 utc_time 时间戳解析，否则当作字符串
            val = row[col]
            try:
                dt = datetime.utcfromtimestamp(int(val)).date()
            except (ValueError, TypeError, OverflowError):
                dt = self._to_date(str(val))
            return dt

        result['_dt'] = result.apply(_flag, axis=1)
        result['holiday'] = result['_dt'].apply(lambda d: d in self._index)
        result['holiday_name'] = result['_dt'].apply(
            lambda d: '; '.join(h.description for h in self._index.get(d, []))
        )
        result['holiday_type'] = result['_dt'].apply(
            lambda d: '; '.join(h.type for h in self._index.get(d, []))
        )
        result.drop(columns=['_dt'], inplace=True)
        return result

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _to_date(dt) -> Date:
        """统一将 datetime / date / str 转为 date"""
        if isinstance(dt, datetime):
            return dt.date()
        if isinstance(dt, Date):
            return dt
        if isinstance(dt, str):
            return Date.fromisoformat(dt.strip())
        raise TypeError(f'Unsupported type: {type(dt)}')


# ------------------------------------------------------------------
# 全局单例（方便快速使用）
# ------------------------------------------------------------------
_calendar: Optional[HolidayCalendar] = None


def _get_calendar() -> HolidayCalendar:
    global _calendar
    if _calendar is None:
        _calendar = HolidayCalendar()
    return _calendar


def is_holiday(dt) -> bool:
    """快捷函数：判断某天是否为节假日"""
    return _get_calendar().is_holiday(dt)


def get_holiday_names(dt) -> list[str]:
    """快捷函数：获取某天的节日名称"""
    return _get_calendar().get_holiday_names(dt)


def merge_all_holiday_records(input_dir: str = '../data/yearly',
                              output_file: str = '../data/all_holiday_records.csv'):
    """
    合并多个 CSV 文件中的节假日记录，提取所有落在节假日的数据。

    参数：
        input_dir: 输入目录（包含多个 data_*.csv 文件）
        output_file: 输出文件路径
    """
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from extrace import load_records_pd

    cal = HolidayCalendar()

    # 获取所有 CSV 文件
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    if not csv_files:
        print(f"警告: {input_dir} 中没有找到 CSV 文件")
        return None

    print(f"发现 {len(csv_files)} 个 CSV 文件")

    all_holiday_records = []
    total_records = 0
    total_holiday = 0

    for csv_file in sorted(csv_files):
        csv_path = os.path.join(input_dir, csv_file)

        try:
            # 加载数据
            df = load_records_pd(csv_path)
            total_records += len(df)

            # 添加节假日标记
            df_tagged = cal.add_holiday_flags(df, col='utc_time')

            # 筛选节假日记录
            holiday_df = df_tagged[df_tagged['holiday']].copy()

            # 添加来源文件信息
            holiday_df['source_file'] = csv_file

            all_holiday_records.append(holiday_df)
            total_holiday += len(holiday_df)

            print(f"✓ {csv_file}: {len(df)} 条记录, {len(holiday_df)} 条节假日")

        except Exception as e:
            print(f"✗ {csv_file}: 处理失败 - {e}")
            continue

    if not all_holiday_records:
        print("未找到任何节假日记录")
        return None

    # 合并所有 DataFrame
    merged_df = pd.concat(all_holiday_records, ignore_index=True)

    # 保存到 CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"\n{'=' * 60}")
    print(f"合并完成！")
    print(f"{'=' * 60}")
    print(f"总记录数:     {total_records}")
    print(f"节假日记录:   {total_holiday}")
    print(f"占比:         {total_holiday / total_records * 100:.2f}%")
    print(f"输出文件:     {output_file}")
    print(f"{'=' * 60}\n")

    # 统计信息
    print("节假日分布统计:")
    holiday_stats = merged_df['holiday_name'].value_counts()
    for name, count in holiday_stats.head(20).items():
        print(f"  {name:20s}: {count:5d} 条")

    return merged_df

def main_test():
    cal = HolidayCalendar()

    # 示例 1：判断单日
    print(f"2022-07-04 是节假日？{cal.is_holiday('2022-07-04')} → {cal.get_holiday_names('2022-07-04')}")
    print(f"2022-07-05 是节假日？{cal.is_holiday('2022-07-05')}")

    # 示例 2：统计节日类型
    from collections import Counter
    type_cnt = Counter(h.type for h in cal.all_holidays)
    print(f"\n节日类型分布: {dict(type_cnt)}")

    # 示例 3：与 extrace.py 的 pd.DataFrame 配合
    print("\n--- 与 DataFrame 配合（按 is_seeker 过滤用户发言） ---")
    data_dir = os.path.join(os.path.dirname(__file__), 'data/yearly')
    csv_candidates = [f for f in os.listdir(data_dir) if f.endswith('.csv') and f != 'holiday.csv']
    for csv_file in csv_candidates:
        csv_path = os.path.join(data_dir, csv_file)
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from extrace import load_records_pd

        df = load_records_pd(csv_path)
        df_tagged = cal.add_holiday_flags(df, col='utc_time')
        print(f"总记录: {len(df_tagged)}, 其中节假日: {df_tagged['holiday'].sum()}")
        # 展示节假日中的用户发言
        user_holiday = df_tagged[df_tagged['holiday'] & (df_tagged['is_seeker'] == True)]
        print(f"用户发言落在节假日的: {len(user_holiday)} 条")
        for _, r in user_holiday.head(5).iterrows():
            d = datetime.utcfromtimestamp(int(r['utc_time'])).date()
            print(f"  {r['conv_id']}  {d} → {cal.get_holiday_names(d)}")

        # 示例 4：直接给 DataFrame 打标签
        import pandas as pd
        df = pd.read_csv(os.path.join(data_dir, csv_file))
        df_tagged = cal.add_holiday_flags(df, col='utc_time')
        holiday_rows = df_tagged[df_tagged['holiday']]
        print(f"\n{len(holiday_rows)} 条记录落在节假日")
        for _, r in holiday_rows.head(5).iterrows():
            print(f"  {r['conv_id']} → {r['holiday_name']} ({r['holiday_type']})")


# ------------------------------------------------------------------
# 使用示例
# ------------------------------------------------------------------
if __name__ == '__main__':
    # 运行合并功能
    result = merge_all_holiday_records()

    # 如果需要查看结果
    if result is not None:
        print(f"\n前 5 条记录:")
        print(result[['conv_id', 'utc_time', 'holiday_name', 'source_file']].head())