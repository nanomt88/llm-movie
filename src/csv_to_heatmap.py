import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.config import (
    OUTPUT_DIR, EMOTION_CATEGORIES,
    sanitize_filename, _setup_font,
    generate_run_id, log, log_error, log_warn,
)

# 1. 解决中文字体显示问题 (如果是Mac，将 'SimHei' 改为 'Arial Unicode MS')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_heatmap(pic_list : list[ str] ):
    for file_path in pic_list:
        # 从列表中提取文件路径和标签
        csv_path = file_path[0]
        x_label = file_path[1]
        y_label = file_path[2]
        pic_title = csv_path[csv_path.index('heatmap')+8 : csv_path.index('.csv')]
        df = pd.read_csv(csv_path, index_col=0)

        # ==========================================
        # 方案一：你的思路 —— 按观影总频次排序
        # ==========================================

        # 计算每一行（情绪）和每一列（节日）的总和，并按降序获取索引
        row_order = df.sum(axis=1).sort_values(ascending=False).index
        col_order = df.sum(axis=0).sort_values(ascending=False).index

        # 根据排序好的索引重排 DataFrame
        df_sorted = df.loc[row_order, col_order]

        # 创建新图形
        fig, ax = plt.subplots(figsize=(14, 6))

        # 绘制基础热力图
        # plt.figure(figsize=(14, 6))
        sns.heatmap(df_sorted, annot=True, fmt="d", cmap="YlOrRd", linewidths=.5)
        # 设置标题和轴标签
        ax.set_title("热力图：观影频次图" + pic_title, fontsize=16, pad=15)
        ax.set_xlabel(x_label, fontsize=12, labelpad=10)
        ax.set_ylabel(y_label, fontsize=12, labelpad=10)
        # [旋转] 横轴标签旋转30度，防止重叠；纵轴保持水平
        ax.tick_params(axis='x', rotation=30)
        ax.tick_params(axis='y', rotation=0)
        # 自动调整布局
        plt.tight_layout()

        fname = csv_path.replace('.csv', '')+'.png'
        out_path = os.path.join(OUTPUT_DIR, fname)
        # 保存图片到文件
        plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"图片已保存: {out_path}")

        # ==========================================
        # 方案二：最优解 —— 层次聚类热力图 (Clustermap)
        # ==========================================

        # seaborn 的 clustermap 会自动对行和列进行聚类计算并重新排序
        # 左侧和上方的“树状图”展示了聚集的过程和相似度距离
        g = sns.clustermap(
            df,
            annot=True,  # 显示具体数值
            fmt="d",  # 数值格式为整数
            cmap="YlOrRd",  # 配色方案：黄-橙-红
            linewidths=.5,  # 单元格间距
            figsize=(14, 8),  # 图表尺寸
            dendrogram_ratio=(0.1, 0.2),  # 树状图所占的比例
            cbar_pos=(1.02, 0.2, 0.03, 0.4)  # 颜色条位置
        )

        g.fig.suptitle("聚类热力图：自动发现相似模式"+pic_title, y=1.05, fontsize=16)
        g.ax_heatmap.set_xlabel(x_label)
        g.ax_heatmap.set_ylabel(y_label)
        # [旋转] 横轴标签旋转30度，防止重叠；纵轴保持水平
        # [旋转] clustermap 需要使用 tick_params 或直接操作 tick labels
        for label in g.ax_heatmap.get_xticklabels():
            label.set_rotation(30)
            label.set_ha('right')

        for label in g.ax_heatmap.get_yticklabels():
            label.set_rotation(0)

        fname = csv_path.replace('.csv', '') + '-2.png'
        out_path = os.path.join(OUTPUT_DIR, fname)
        # 保存图片到文件
        plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"图片已保存: {out_path}")
        #plt.show()
        # 可选：关闭图形释放内存
        plt.close()


def main():
    # 2. 读取 CSV 文件（将第一列情绪设为行索引）
    # 请确保文件名与你实际的文件路径一致
    plot_heatmap([
                  ['../output/20260607_111642_S05_heatmap1_emotion_holiday_18-25.csv', 'holiday', 'emotion'],
                  ['../output/20260607_111642_S05_heatmap1_emotion_holiday_26-35.csv', 'holiday', 'emotion'],
                  ['../output/20260607_111642_S05_heatmap1_emotion_holiday_36-50.csv', 'holiday', 'emotion'],
                  ['../output/20260607_111642_S05_heatmap1_emotion_holiday_50p.csv', 'holiday', 'emotion'],
                  ['../output/20260607_111642_S05_heatmap1_emotion_holiday_lt18.csv', 'holiday', 'emotion'],
                  ['../output/20260607_111642_S05_heatmap1_emotion_holiday_unknown.csv', 'holiday', 'emotion'],
                  ['../output/20260607_111642_S05_heatmap2_emotion_genre_18-25.csv', 'genre', 'emotion'],
                  ['../output/20260607_111642_S05_heatmap2_emotion_genre_26-35.csv','genre', 'emotion'],
                  ['../output/20260607_111642_S05_heatmap2_emotion_genre_36-50.csv','genre', 'emotion'],
                  ['../output/20260607_111642_S05_heatmap2_emotion_genre_50p.csv','genre', 'emotion'],
                  ['../output/20260607_111642_S05_heatmap2_emotion_genre_lt18.csv','genre', 'emotion'],
                  ['../output/20260607_111642_S05_heatmap2_emotion_genre_unknown.csv','genre', 'emotion'],
                  ['../output/20260607_111642_S05_heatmap3_holiday_genre_18-25.csv', 'holiday', 'genre'],
                  ['../output/20260607_111642_S05_heatmap3_holiday_genre_26-35.csv','holiday', 'genre'],
                  ['../output/20260607_111642_S05_heatmap3_holiday_genre_36-50.csv','holiday', 'genre'],
                  ['../output/20260607_111642_S05_heatmap3_holiday_genre_50p.csv','holiday', 'genre'],
                  ['../output/20260607_111642_S05_heatmap3_holiday_genre_lt18.csv','holiday', 'genre'],
                  ['../output/20260607_111642_S05_heatmap3_holiday_genre_unknown.csv','holiday', 'genre'],
    ])

if __name__ == '__main__':
    main()