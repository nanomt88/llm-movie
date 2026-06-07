import json

def extract_movie_genres(file_path):
    """
    从JSON文件中提取所有电影的类型(genres)

    Args:
        file_path: JSON文件路径

    Returns:
        包含所有电影类型的列表
    """
    # 读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as f:
        movie_data = json.load(f)

    # 存储所有电影类型
    all_genres = []

    # 遍历每部电影
    for movie_id, movie_info in movie_data.items():
        if 'genres' in movie_info:
            genres = movie_info['genres']
            all_genres.extend(genres)

    return all_genres

def get_unique_genres(genres_list):
    """
    获取不重复的电影类型

    Args:
        genres_list: 电影类型列表

    Returns:
        不重复的电影类型集合
    """
    return list(set(genres_list))

# 使用示例
if __name__ == "__main__":
    file_path = "../data/movie_info.json"

    # 提取所有电影类型
    genres = extract_movie_genres(file_path)

    #print("所有电影类型:")
    #print(genres)
    print(f"\n总共 {len(genres)} 个类型记录")

    # 获取不重复的类型
    unique_genres = get_unique_genres(genres)
    print("\n不重复的电影类型:")
    print(sorted(unique_genres))
    print(f"\n共有 {len(unique_genres)} 种不同的类型")

    # 统计每种类型出现的次数
    from collections import Counter
    genre_counts = Counter(genres)
    print("\n各类型出现次数:")
    for genre, count in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{genre}: {count}")
