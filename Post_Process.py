import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import seaborn as sns

def printPlt(data_path,lab,shape):
    df = pd.read_csv(data_path,encoding='ANSI')
# 获取预测与真实值（已为 log10 值）
    prediction = df['Prediction'].values
    experiment = df['Truth'].values

# 计算误差和指标
    error = experiment - prediction
    mae = mean_absolute_error(experiment, prediction)
    mse = mean_squared_error(experiment, prediction)
    r2 = r2_score(experiment, prediction)

# 2× Scatter Band 边界（对数空间中 ±log10(2)）
    log2 = np.log10(2)

# 绘图
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)

# 散点图
    sc = ax.scatter(prediction, experiment, 
                c=np.abs(error),
                cmap='viridis',
                alpha=0.7,
                edgecolors='w',
                linewidths=0.5,
                s=60,
                marker=shape,
                label=f'{lab} Data points (n={len(df)})')

    cbar = plt.colorbar(sc)
    cbar.set_label('Absolute Error', rotation=270, labelpad=15)

# 拟合线
    min_val = min(prediction.min(), experiment.min())
    max_val = max(prediction.max(), experiment.max())
    x_range = np.linspace(min_val, max_val, 100)
    ax.plot(x_range, x_range, '--', color='#FF2D2D', lw=2, label='Perfect prediction')

# ±log(2) scatter band（对数空间中加减）
    ax.fill_between(x_range, x_range - log2, x_range + log2,
                color='#FFA500', alpha=0.2, label='2× scatter band')

# 统计在 2× scatter band 内的点比例
    within_2x = np.mean(np.abs(prediction - experiment) <= log2)

# 信息框
    stats_text = f"""
    Model Performance:
    R² = {r2:.4f}
    MAE = {mae:.4f}
    MSE = {mse:.4f}
    Within 2× scatter band = {within_2x * 100:.1f}%
    """
    ax.text(0.95, 0.05, stats_text,
        transform=ax.transAxes,
        ha='right', va='bottom',
        bbox=dict(facecolor='white', alpha=0.8))

# 标签与图例
    ax.set_xlabel('Predicted Fatigue Life ', fontsize=12)
    ax.set_ylabel('Experimental Fatigue Life', fontsize=12)
    ax.set_title(f"{lab} Tranining Prediction vs Experimental Results", fontsize=14, pad=20)
    ax.legend(loc='upper left', framealpha=1)
    plt.tight_layout()

# 保存图像
    output_path = f"{lab}_scatter_plot.png"  
    plt.savefig(output_path, dpi=300, bbox_inches='tight') 
    print(f"图像已保存为 {output_path}")
    plt.show()

def printPltAll(data_path1,data_path2,data_path3):
    df1 = pd.read_csv(data_path1,encoding='ANSI')
    df2 = pd.read_csv(data_path2,encoding='ANSI')
    df3 = pd.read_csv(data_path3,encoding='ANSI')
# 获取预测与真实值（已为 log10 值）
    prediction1 = df1['Prediction'].values
    experiment1 = df1['Truth'].values
    prediction2 = df2['Prediction'].values
    experiment2 = df2['Truth'].values
    prediction3 = df3['Prediction'].values
    experiment3 = df3['Truth'].values

# 计算误差和指标
    mae = mean_absolute_error(experiment3, prediction3)
    mse = mean_squared_error(experiment3, prediction3)
    r2 = r2_score(experiment3, prediction3)

# 2× Scatter Band 边界（对数空间中 ±log10(2)）
    log2 = np.log10(2)

# 绘图
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)

# 散点图
    scr = ax.scatter(prediction1, experiment1, 
                c=np.abs(experiment1 - prediction1),
                cmap='viridis',
                alpha=0.7,
                edgecolors='w',
                linewidths=0.5,
                s=60,
                marker='o',
                label=f'real Data points (n={len(df1)})')
    scs = ax.scatter(prediction2, experiment2, 
                c=np.abs(experiment2 - prediction2),
                cmap='viridis',
                alpha=0.7,
                edgecolors='w',
                linewidths=0.5,
                s=60,
                marker='s',
                label=f'fake Data points (n={len(df2)})')
    cbar = plt.colorbar(scr)
    # scbar = plt.colorbar(scs)
    cbar.set_label('Absolute Error', rotation=270, labelpad=15)
    # scbar.set_label('Absolute Error', rotation=270, labelpad=15)

# 拟合线
    min_val = min(prediction3.min(), experiment3.min())
    max_val = max(prediction3.max(), experiment3.max())
    x_range = np.linspace(min_val, max_val, 100)
    ax.plot(x_range, x_range, '--', color='#FF2D2D', lw=2, label='Perfect prediction')

# ±log(2) scatter band（对数空间中加减）
    ax.fill_between(x_range, x_range - log2, x_range + log2,
                color='#FFA500', alpha=0.2, label='2× scatter band')

# 统计在 2× scatter band 内的点比例
    within_2x = np.mean(np.abs(prediction3 - experiment3) <= log2)

# 信息框
    stats_text = f"""
    Model Performance:
    R² = {r2:.4f}
    MAE = {mae:.4f}
    MSE = {mse:.4f}
    Within 2× scatter band = {within_2x * 100:.1f}%
    """
    ax.text(0.95, 0.05, stats_text,
        transform=ax.transAxes,
        ha='right', va='bottom',
        bbox=dict(facecolor='white', alpha=0.8))

# 标签与图例
    ax.set_xlabel('Predicted Fatigue Life', fontsize=12)
    ax.set_ylabel('Experimental Fatigue Life', fontsize=12)
    ax.set_title('ALL Tranining Prediction vs Experimental Results', fontsize=14, pad=20)
    ax.legend(loc='upper left', framealpha=1)
    plt.tight_layout()

# 保存图像
    output_path = f"all_scatter_plot.png"  
    plt.savefig(output_path, dpi=300, bbox_inches='tight') 
    print(f"图像已保存为 {output_path}")
    plt.show()

data_p = "real_data.csv"
data_s = "samp_data.csv"
data_a = "samp_real.csv"

# 设置样式
sns.set_style('whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

printPlt(data_p,"Real",'o')
printPlt(data_s,"Sample",'s')
printPltAll(data_p,data_s,data_a)