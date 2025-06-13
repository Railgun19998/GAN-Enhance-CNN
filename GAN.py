import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from math import sqrt

#PCA可视化
def visualize_pca(generator, real_data, latent_dim, epoch, num_samples=500):
    # 生成伪造数据
    noise = np.random.normal(0, 1, (num_samples, latent_dim))
    generated_data = generator.predict(noise, verbose=0)

    # 展平真实数据和生成数据
    real_data_flatten = real_data.reshape(real_data.shape[0], -1)
    generated_data_flatten = generated_data.reshape(generated_data.shape[0], -1)

    # 合并数据
    combined_data = np.vstack([real_data_flatten, generated_data_flatten])
    labels = np.array([0] * real_data_flatten.shape[0] + [1] * generated_data_flatten.shape[0])  # 0 表示真实数据，1 表示生成数据

    # 使用 PCA 进行降维
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(combined_data)

    # 绘制分布图
    plt.figure(figsize=(8, 8))
    plt.scatter(reduced_data[labels == 0][:, 0], reduced_data[labels == 0][:, 1], alpha=0.5, label='Real Data', color='blue')
    plt.scatter(reduced_data[labels == 1][:, 0], reduced_data[labels == 1][:, 1], alpha=0.5, label='Generated Data', color='orange')
    plt.title(f'PCA Visualization at Epoch {epoch}')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.grid(True)
    # plt.show()
    # 保存图像
    save_dir = "pca_visualizations"  # Define the directory for saving PCA visualizations
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
    save_path = os.path.join(save_dir, f"pca_epoch_{epoch}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()  # 关闭图像，释放内存

#损失曲线绘制函数
def printLoss(loss_history,lab):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label=f'Discriminator Loss ({lab} Loss)', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'GAN Training {lab}Loss')
    plt.legend()
    plt.grid(True)
    save_dir = "training_visualizations"  # 定义保存图像的目录
    os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在，则创建
    save_path = os.path.join(save_dir, f"{lab}_training_loss.png")  # 图像保存路径
    plt.savefig(save_path, dpi=300)  # 保存图像，设置分辨率为 300 DPI
    plt.close()  # 关闭图像，释放内存

# ========================================
# 1. 构建GAN模型（输入输出均为25+1维）
# ========================================
# 生成器定义
def build_generator():
    model = models.Sequential([
        layers.Dense(512, input_dim=latent_dim, activation='leaky_relu', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='leaky_relu', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='leaky_relu', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Dense(25 + 1, activation='tanh')  # 输出层保持 tanh
    ])
    return model

# 判别器定义
def build_discriminator():
    model = models.Sequential([
        layers.GaussianNoise(0.1, input_shape=(25+1,)),  # 降低噪声强度
        layers.Dense(1024, activation='leaky_relu', kernel_initializer='he_normal'),
        layers.Dropout(0.2),  # 降低Dropout比例
        layers.Dense(512, activation='leaky_relu', kernel_initializer='he_normal'),
        layers.Dropout(0.2),
        layers.Dense(256, activation='leaky_relu', kernel_initializer='he_normal'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# 训练定义
def train(epochs,batch_size):
    d_loss_history = []
    g_loss_history = []
    d_loss = 1.0
    g_loss = 0
    for epoch in range(epochs):
        # 训练判别器
        d_loss_history.append(d_loss)
        if d_loss > 0.55 or g_loss < 0.40:  # 仅当判别器不够强时更新
            real_labels = np.random.uniform(0.95, 1.0, (batch_size, 1))
            fake_labels = np.random.uniform(0.0, 0.05, (batch_size, 1))
            for _ in range(5):
                discriminator.trainable = True  # 解冻判别器
                idx = np.random.randint(0, len(real_data), batch_size)
                real_batch = real_data[idx]
    
                noise = np.random.normal(0, 1, (batch_size, latent_dim))
                fake_batch = generator.predict(noise,verbose=0)
                d_loss_real = discriminator.train_on_batch(real_batch, real_labels)
                d_loss_fake = discriminator.train_on_batch(fake_batch, fake_labels)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                d_loss_history.append(d_loss)
                
        # 训练生成器
        discriminator.trainable = False  # 冻判别器
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
        g_loss_history.append(g_loss)

    # 每100 epoch打印进度
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | D Loss: {d_loss:.4f} | G Loss: {g_loss:.4f}")
        if epoch % (epochs / 20) == 0:    
            visualize_pca(generator, real_data, latent_dim, epoch, num_samples=500)

    # 每隔 1000 个 epoch 保存一次生成器模型
        if epoch % 500 == 0 and epoch > 0:
            generator.save(f'generator_epoch_{epoch}.h5')
            print(f"Generator model saved at epoch {epoch}")

    # 绘制损失曲线
    printLoss(d_loss_history,"D")
    printLoss(g_loss_history,"G")

    # 保存生成器模型
    generator.save('generator_model.h5') 

# ========================================
# 2. 数据预处理（使用原始25维特征）
# ========================================
data = pd.read_csv("data.S3.csv",encoding='ANSI') # 加载数据
data = data.iloc[:, 1:]  # 去掉第一列索引
features = data.drop('Creep rupture life (h)', axis=1).values  # 原始25维特征
target = data['Creep rupture life (h)'].values

scaler = MinMaxScaler(feature_range=(-1, 1))  # 标准化（仅对原始25维特征）
scaled_data = scaler.fit_transform(np.hstack([features, target.reshape(-1,1)]))  # 合并成26维
real_data = scaled_data

# ========================================
# 3. 训练GAN（使用25维数据）
# ========================================
latent_dim = 100 #虚假维度
generator = build_generator()
discriminator = build_discriminator()
discriminator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0004, 
    beta_1=0.5,  # 动量参数调整
    clipvalue=0.5  # 梯度裁剪
)
generator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0001,
    beta_1=0.5
)
discriminator.compile(optimizer=discriminator_optimizer,loss='binary_crossentropy')
discriminator.trainable = False

gan = models.Sequential([generator, discriminator])
gan.compile(optimizer=generator_optimizer, loss='binary_crossentropy')

#train(epochs = 8000, batch_size = 800) # 训练GAN

# ========================================
# 4. 生成数据并填充至36维
# ========================================
gen_model = load_model("generator_model.h5")

num_samples = 800 # 生成样本数
noise = np.random.normal(0, 1, (num_samples, latent_dim))

generated_samples = gen_model.predict(noise) # 生成样本并反标准化
gen_features = scaler.inverse_transform(generated_samples[:, :26])

gen_features_25d = gen_features[:, :25]  # 生成的25维特征
gen_creep_life = gen_features[:, -1]  # 生成的Creep rupture life (h)
gen_target = np.log10(gen_features[:, -1]) # 生成的Creep rupture life (h)

gen_features_df = pd.DataFrame(
    gen_features, 
    columns=[f'Feature_{i+1}' for i in range(gen_features.shape[1])]
)

gen_features_df.to_csv('generated_features.csv', index=False) # 保存为 CSV 文件

# ========================================
# 5. CNN模型定义与预测
# ========================================
cnn_model = load_model("cnn_Ti.h5") # 加载预训练CNN模型

padded_scaler = MinMaxScaler() # 标准化生成数据
X_all_scaler = padded_scaler.fit_transform(gen_features_25d)
x_data =[]
for i in range(len(X_all_scaler)):
    a = X_all_scaler[i,:]
    a = np.pad(a,(0,11),'constant',constant_values=(0,0))
    a = a.reshape(6,6,1)
    x_data.append(a)
x_data =np.array(x_data)
cnn_pred = cnn_model.predict(x_data).flatten() # 预测

mse = np.sum((gen_target - cnn_pred) ** 2) / len(gen_target)
rmse = sqrt(mse)
mae = np.sum(np.absolute(gen_target - cnn_pred)) / len(gen_target)
r2 = 1-mse/ np.var(gen_target)
print(" mae:",mae,"mse:",mse," rmse:",rmse," r2:",r2)

# ========================================
# 6. 剔除数据
# ========================================
absolute_error = np.abs(gen_target - cnn_pred)
threshold = 0.6
valid_indices = np.where(absolute_error <= threshold)[0]
filtered_gen_features_25d = gen_features_25d[valid_indices]
filtered_gen_target = gen_creep_life[valid_indices]
filtered_cnn_pred = cnn_pred[valid_indices]
print(f"剔除 {len(gen_target) - len(filtered_gen_target)} 条数据，剩余 {len(filtered_gen_target)} 条数据。")

# 将剔除后的数据集保存为 CSV 文件
filtered_data = pd.DataFrame(
    np.hstack([filtered_gen_features_25d, filtered_gen_target.reshape(-1, 1)]),
    columns=[f'Feature_{i+1}' for i in range(filtered_gen_features_25d.shape[1])] + ['Target']
)
filtered_data.to_csv('filtered_generated_features.csv', index=False)

mse = np.sum((np.log10(filtered_gen_target) - filtered_cnn_pred) ** 2) / len(filtered_gen_target)
rmse = sqrt(mse)
mae = np.sum(np.absolute(np.log10(filtered_gen_target) - filtered_cnn_pred)) / len(filtered_gen_target)
r2 = 1-mse/ np.var(filtered_gen_target)#均方误差/方差
print("剔除后的统计数据 mae:",mae,"mse:",mse," rmse:",rmse," r2:",r2)

# ========================================
# 7. 结果可视化
# ========================================
plt.figure(figsize=(10, 6))
# plt.scatter(gen_target, cnn_pred, alpha=0.6, c='blue', edgecolor='k') #原生成数据散点
plt.scatter(np.log10(filtered_gen_target), filtered_cnn_pred, alpha=0.6, c='blue', edgecolor='k') #剔除后数据散点
plt.plot([0, 4], [0, 4], 'r--', linewidth=2)
plt.xlabel('GAN Generated Ti_lg_RT')
plt.ylabel('CNN Predicted Ti_lg_RT')
plt.title('GAN ->CNN Prediction Comparison')
plt.grid(True)
plt.text(-2.8,6, f"MAE: {mae:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\nR2: {r2:.4f}",
         fontsize=10, fontproperties='Times New Roman',
         bbox=dict(facecolor='white', alpha=0.5))
plt.savefig('Superalloys-Ti_CNN_lgRT.png',dpi=500, bbox_inches='tight')
plt.show()