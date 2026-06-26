import os
# 必须在导入 datasets 之前设置镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from datasets import load_dataset
import pandas as pd


def download_and_split_streaming(dataset_name: str = "ZhankuiHe/reddit_movie_large_v1",
                                  output_dir: str = 'data/yearly'):
    """
    使用流式加载处理大数据集，避免内存溢出
    """
    os.makedirs(output_dir, exist_ok=True)

    # 流式加载
    print(f"正在流式加载数据集: {dataset_name}")
    ds = load_dataset(dataset_name, streaming=True)

    # 准备每年的 DataFrame
    yearly_data = {year: [] for year in range(2010, 2023)}

    # 遍历数据
    count = 0
    for example in ds['validation']:
        try:
            utc_time = example.get('utc_time', 0)
            # 如果是字符串，转为数字
            if isinstance(utc_time, str):
                utc_time = int(utc_time)

            # 10位时间戳（秒级），转换为毫秒后解析
            dt = pd.to_datetime(utc_time, unit='s')

            year = dt.year

            if 2010 <= year <= 2022:
                yearly_data[year].append(example)
                count += 1

                if count % 10000 == 0:
                    print(f"已处理 {count} 条数据...")
        except Exception as e:
            continue

    # 保存每年的数据
    for year in range(2010, 2023):
        data = yearly_data[year]
        if data:
            df = pd.DataFrame(data)
            filepath = os.path.join(output_dir, f'data_validation_{year}.csv')
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"✓ {year}年: {len(df)} 条数据 -> {filepath}")
        else:
            print(f"✗ {year}年: 无数据")

    print(f"\n完成！")



if __name__ == '__main__':
    download_and_split_streaming()
