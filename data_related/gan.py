import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler


class GANImputer:
    def __init__(self, df):
        self.df = df
        self.scaler = MinMaxScaler()
        self.latent_dim = 100
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns
        self.mask = None
        self.df_normalized = None
        self.generator = None

    def preprocess(self):
        """预处理数据并创建缺失值掩码"""
        print("正在预处理数据...")
        # 归一化数值列
        self.df_normalized = self.df.copy()
        self.df_normalized[self.numeric_cols] = self.scaler.fit_transform(
            self.df[self.numeric_cols].fillna(self.df[self.numeric_cols].mean())
        )

        # 创建缺失值掩码
        self.mask = self.df[self.numeric_cols].isna()
        print(f"发现缺失值总数: {self.mask.sum().sum()}")

        return self.df_normalized

    def build_generator(self, input_dim):
        """构建生成器模型"""
        noise_input = layers.Input(shape=(self.latent_dim,))

        # 将噪声输入转换为所需维度
        x = layers.Dense(256)(noise_input)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Dense(512)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Dense(256)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.BatchNormalization()(x)

        # 输出层的维度应该与特征数量相同
        outputs = layers.Dense(input_dim, activation='tanh')(x)

        return tf.keras.Model(noise_input, outputs)

    def build_discriminator(self, input_dim):
        """构建判别器模型"""
        model = tf.keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(256),
            layers.LeakyReLU(alpha=0.2),
            layers.Dense(128),
            layers.LeakyReLU(alpha=0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        return model

    def train_gan(self, epochs=1000, batch_size=32):
        """训练GAN模型"""
        print("\n开始训练GAN模型...")

        # 获取特征维度
        input_dim = len(self.numeric_cols)
        print(f"特征维度: {input_dim}")

        # 构建生成器和判别器
        self.generator = self.build_generator(input_dim)
        discriminator = self.build_discriminator(input_dim)

        # 编译模型
        discriminator.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            loss='binary_crossentropy'
        )

        # 构建并编译完整的GAN
        discriminator.trainable = False
        gan_input = layers.Input(shape=(self.latent_dim,))
        gan_output = discriminator(self.generator(gan_input))
        gan = tf.keras.Model(gan_input, gan_output)
        gan.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            loss='binary_crossentropy'
        )

        # 获取非缺失值的完整样本用于训练
        complete_samples = self.df_normalized[self.numeric_cols][~self.mask.any(axis=1)]

        if len(complete_samples) < batch_size:
            print("警告：完整样本数量少于批次大小，调整批次大小")
            batch_size = len(complete_samples)

        # 训练循环
        for epoch in range(epochs):
            # 训练判别器
            idx = np.random.randint(0, len(complete_samples), batch_size)
            real_batch = complete_samples.iloc[idx].values

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            generated_data = self.generator.predict(noise, verbose=0)

            d_loss_real = discriminator.train_on_batch(real_batch, np.ones((batch_size, 1)))
            d_loss_fake = discriminator.train_on_batch(generated_data, np.zeros((batch_size, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # 训练生成器
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs}, D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")

        print("GAN模型训练完成！")
        return self.generator

    def impute_missing_values(self):
        """使用训练好的GAN填充缺失值"""
        print("\n开始填充缺失值...")

        # 如果还没有预处理，先进行预处理
        if self.mask is None:
            self.preprocess()

        # 如果还没有训练GAN，先训练模型
        if self.generator is None:
            self.train_gan()

        # 获取有缺失值的行
        missing_rows = self.df[self.mask.any(axis=1)].index

        if len(missing_rows) > 0:
            print(f"正在处理 {len(missing_rows)} 行缺失数据...")

            # 为每个缺失值生成填充值
            noise = np.random.normal(0, 1, (len(missing_rows), self.latent_dim))
            generated_values = self.generator.predict(noise, verbose=0)

            # 将生成的值填充到原始数据中
            imputed_df = self.df.copy()

            # 只填充缺失的值
            for i, row_idx in enumerate(missing_rows):
                for j, col in enumerate(self.numeric_cols):
                    if self.mask.loc[row_idx, col]:
                        imputed_df.loc[row_idx, col] = self.scaler.inverse_transform(
                            generated_values[i].reshape(1, -1)
                        )[0, j]

            print(f"成功填充 {len(missing_rows)} 行的缺失值")
            return imputed_df
        else:
            print("没有发现缺失值，返回原始数据")
            return self.df


if __name__ == "__main__":
    # 读取数据
    print("正在读取数据...")
    df = pd.read_csv('data_cleaned.csv')

    # 创建GANImputer实例
    imputer = GANImputer(df)

    # 填充缺失值
    df_imputed = imputer.impute_missing_values()

    # 保存处理后的数据
    df_imputed.to_csv('data_imputed.csv', index=False)
    print("\n数据已保存至 'data_imputed.csv'")