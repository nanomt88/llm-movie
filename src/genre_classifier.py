# -*- coding: utf-8 -*-
"""
电影类型分类模块 — 从 movie_info.json 中按 IMDB ID 查找电影类型。

无文件输出（纯工具函数），但提取过程中会将未匹配的 IMDB ID 记录到日志。
"""

from src.config import log, log_error, log_warn


def get_genres_from_movie_info(imdb_id: str,
                               movie_info: dict,
                               not_found_log: list[str]) -> list[str] | None:
    """
    从 movie_info 中按 IMDB ID 查找电影类型。

    参数：
        imdb_id       : IMDB ID (如 'tt0111161')
        movie_info    : movie_info.json 加载后的 dict
        not_found_log : 未找到的 IMDB ID 记录列表（引用传递）

    返回：
        类型列表（如 ['剧情', '犯罪']），未找到返回 None
    """
    info = movie_info.get(imdb_id)
    if info is None:
        not_found_log.append(imdb_id)
        return None
    genres = info.get('genres', [])
    return genres


def classify_genre_from_ids(imdb_ids: list[str],
                            movie_info: dict,
                            not_found_log: list[str]) -> list[str]:
    """
    对一组 IMDB ID 查找电影类型，合并去重。

    参数：
        imdb_ids      : IMDB ID 列表
        movie_info    : movie_info.json 加载后的 dict
        not_found_log : 未找到的 IMDB ID 记录列表

    返回：
        合并去重后的类型列表（如 ['剧情', '犯罪']），无匹配时返回 ['剧情']
    """
    all_genres = set()
    for imdb_id in imdb_ids:
        genres = get_genres_from_movie_info(imdb_id, movie_info, not_found_log)
        if genres is not None:
            all_genres.update(genres)
    return list(all_genres) if all_genres else ['剧情']


def log_not_found_ids(not_found_ids: list[str]):
    """
    将未匹配的 IMDB ID 输出到日志（最多显示 20 个避免刷屏）。
    """
    if not not_found_ids:
        return
    unique = sorted(set(not_found_ids))
    count = len(unique)
    log('Genre', f'[!] {count} 个 IMDB ID 在 movie_info.json 中未找到:')

    # 控制台只打印前 20 个
    for tid in unique[:20]:
        print(f"      - {tid}", end=", ")
    if count > 20:
        print(f"... 共 {count} 个")
    else:
        print()
