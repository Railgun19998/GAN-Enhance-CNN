import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers,models
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from math import sqrt

def printPlt(prediction1,experiment1,prediction2,experiment2,sw):
# 计算误差和指标
    prediction3 = np.concatenate((prediction1, prediction2))
    experiment3 = np.concatenate((experiment1, experiment2))
    mae = mean_absolute_error(experiment3, prediction3)
    mse = mean_squared_error(experiment3, prediction3)
    r2 = r2_score(experiment3, prediction3)

# 2× Scatter Band 边界（对数空间中 ±log10(2)）
    log2 = np.log10(2)

# 绘图
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)

# 散点图
    if sw == 1:
        mae = mean_absolute_error(experiment2, prediction2)
        mse = mean_squared_error(experiment2, prediction2)
        r2 = r2_score(experiment2, prediction2)
        scr = ax.scatter(prediction1, experiment1, 
                c=np.abs(experiment1 - prediction1),
                cmap='viridis',
                alpha=0.7,
                edgecolors='w',
                linewidths=0.5,
                s=60,
                marker='o',
                label=f'real Data points (n={len(prediction1)})')
        scs = ax.scatter(prediction2, experiment2, 
                c=np.abs(experiment2 - prediction2),
                cmap='viridis',
                alpha=0.7,
                edgecolors='w',
                linewidths=0.5,
                s=60,
                marker='s',
                label=f'fake Data points (n={len(prediction2)})')
    if sw == 0:
        scr = ax.scatter(prediction3, experiment3, 
                c=np.abs(experiment3 - prediction3),
                cmap='viridis',
                alpha=0.7,
                edgecolors='w',
                linewidths=0.5,
                s=60,
                marker='o',
                label=f'real Data points (n={len(prediction3)})')
    cbar = plt.colorbar(scr)
    cbar.set_label('Absolute Error', rotation=270, labelpad=15)

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
    # output_path = f"all_scatter_plot.png"  
    # plt.savefig(output_path, dpi=300, bbox_inches='tight') 
    # print(f"图像已保存为 {output_path}")
    plt.show()

data = pd.read_csv("all_data.csv",encoding='ANSI')
data = data.drop(['Class'],axis=1)

x = data.iloc[:, :-1]
y = np.log10(data.iloc[:, -1])
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

x_data =[]
for i in range(len(x)):
    a = x[i,:]
    a = np.pad(a,(0,11),'constant',constant_values=(0,0))
    a = a.reshape(6,6,1)
    x_data.append(a)
x_data =np.array(x_data)

x_train, x_test, y_train, y_test = train_test_split(x_data, y,train_size=0.8)

#CNN MODEL BUILD
model = models.Sequential()
model.add(layers.Conv2D(8, (2, 2), activation='relu',padding='same', input_shape=(6, 6, 1)))
model.add(layers.Conv2D(16, (2, 2), padding='same',activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1))
model.summary()

adam = optimizers.Adam(learning_rate=0.005)
model.compile(optimizer=adam, loss='mean_squared_error')
checkpoint = ModelCheckpoint(filepath="cnn_sr1.h5", monitor='val_loss', mode='min', save_best_only=True, verbose=1)
#model.fit(x_train, y_train, epochs=1000, batch_size=8,callbacks=[checkpoint],validation_data=[x_test,y_test])

m = load_model("cnn_sr1.h5")
pre_test = m.predict(x_test).flatten()
pre_train = m.predict(x_train).flatten()

# 保存预测结果和真实值到 CSV 文件
# results = pd.DataFrame({
#     'Prediction': pre_source,
#     'Truth': y
# })

# # 保存到文件
# results.to_csv('results_superalloy.csv', index=False)

printPlt(pre_train,y_train,pre_test,y_test,1)
