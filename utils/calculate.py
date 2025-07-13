# 文件名: analyze_log.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def analyze_performance_log(csv_path: str):

    print(f"--- 开始分析日志文件: {csv_path} ---")
    try:
        df = pd.read_csv(csv_path)
        print(f"成功读取 {len(df)} 条数据。")
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{csv_path}'。请检查路径是否正确。")
        return
    # --- 2. 计算统计数据 ---
    # 定义需要分析的列
    columns_to_analyze = [
        'frame_index',
        'proc_time_ms',
        'dynamic_points_count',
        'static_points_count',
        'percent'
    ]
    # 确保所有需要的列都存在
    for col in columns_to_analyze:
        if col not in df.columns:
            print(f"警告: CSV文件中缺少列 '{col}'，将跳过对该列的分析。")
            columns_to_analyze.remove(col)

    if not columns_to_analyze:
        print("错误: 文件中没有可供分析的数据列。")
        return

    # 计算平均值、方差和80%分位数
    mean_values = df[columns_to_analyze].mean()
    variance_values = df[columns_to_analyze].var()
    percentile_80 = df[columns_to_analyze].quantile(0.80)

    # 将结果整合到一个DataFrame中以便清晰展示
    results_df = pd.DataFrame({
        'Mean': mean_values,
        'Variance': variance_values,
        '80th Percentile': percentile_80
    })

    print("\n--- 数据统计结果 ---")
    print(results_df)
    # --- 3. 绘制箱线图 ---
    print("\n正在生成箱线图...")

    # 设置matplotlib以支持中文显示，避免乱码
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 'SimHei' 是一个常用的中文字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    except Exception:
        print("警告: 未找到 'SimHei' 字体，绘图中的中文可能显示为方框。")

    # 我们主要关注性能相关的列，frame_index通常不需要画图
    plot_columns = [
        'proc_time_ms',
        'dynamic_points_count',
        'static_points_count',
        'percent'
    ]
    # 过滤掉不存在的列
    plot_columns = [col for col in plot_columns if col in df.columns]

    if not plot_columns:
        print("没有可供绘图的数据列。")
        return
        
    # 创建一个2x2的子图网格
    num_plots = len(plot_columns)
    num_cols = 2
    num_rows = (num_plots + 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, 6 * num_rows))
    # 将axes数组展平，方便迭代
    axes = axes.flatten()

    for i, col_name in enumerate(plot_columns):
        sns.boxplot(y=df[col_name], ax=axes[i], orient='v')
        axes[i].set_title(f'"{col_name}" box plot', fontsize=14)
        axes[i].set_ylabel("Values")
        axes[i].grid(True, linestyle='--', alpha=0.6)
    
    # 如果子图数量是奇数，隐藏最后一个空的子图
    if num_plots < len(axes):
        for i in range(num_plots, len(axes)):
            fig.delaxes(axes[i])

    # 调整布局并显示图像
    fig.suptitle('log', fontsize=18, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == '__main__':
    # --- 使用示例 ---
    # 请将这里的路径替换为您自己的CSV文件路径
    # 例如: './logs/performance_log_20250701_152553.csv'
    csv_file_path = './logs/performance_log_20250710_185603.csv'
    
    analyze_performance_log(csv_file_path)