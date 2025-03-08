import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler


class RainfallFeatureEngineer:
    def __init__(self, df):
        self.df = df
        # 选择最相关的气象要素
        self.key_features = [
            'T',  # 温度
            'U',  # 相对湿度
            'P',  # 气压
            'ff10',  # 风速
            'prcp'  # 降水量(目标变量)
        ]
        self.scaler = MinMaxScaler()

    def preprocess_features(self):
        """预处理关键特征"""
        # 确保所有关键特征存在
        self.key_features = [f for f in self.key_features if f in self.df.columns]

        # 标准化关键特征
        self.df[self.key_features] = self.scaler.fit_transform(self.df[self.key_features])

    def create_time_features(self):
        """创建时间特征，关注降水的周期性模式"""
        # 基础时间特征
        self.df['month'] = self.df['date'].dt.month
        self.df['day_of_year'] = self.df['date'].dt.dayofyear

        # 降水的季节性特征（使用正弦和余弦变换）
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)
        self.df['day_sin'] = np.sin(2 * np.pi * self.df['day_of_year'] / 365)
        self.df['day_cos'] = np.cos(2 * np.pi * self.df['day_of_year'] / 365)

        # 季节特征
        self.df['season'] = pd.cut(self.df['month'],
                                   bins=[0, 3, 6, 9, 12],
                                   labels=['冬季', '春季', '夏季', '秋季'])

    def create_sequence_features(self, window_sizes=[3, 7, 14]):
        """创建序列特征，用于LSTM的输入"""
        for feature in self.key_features:
            for window in window_sizes:
                # 滑动统计量
                self.df[f'{feature}_rolling_mean_{window}'] = self.df[feature].rolling(window=window).mean()
                self.df[f'{feature}_rolling_std_{window}'] = self.df[feature].rolling(window=window).std()

                # 趋势特征
                self.df[f'{feature}_trend_{window}'] = (self.df[f'{feature}_rolling_mean_{window}'] -
                                                        self.df[feature].rolling(window=window * 2).mean())

                # 与历史的差异
                if feature == 'prcp':
                    # 计算历史同期降水量的统计特征
                    self.df[f'{feature}_historical_mean_{window}'] = self.df.groupby(['month', 'day_of_year'])[
                        feature].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean())
                    self.df[f'{feature}_historical_diff_{window}'] = (self.df[feature] -
                                                                      self.df[f'{feature}_historical_mean_{window}'])

    def create_extreme_rainfall_features(self):
        """创建极端降水相关特征"""
        if 'prcp' in self.df.columns:
            # 定义极端降水阈值（使用分位数）
            extreme_threshold = self.df['prcp'].quantile(0.95)

            # 极端降水标记
            self.df['is_extreme_rain'] = (self.df['prcp'] > extreme_threshold).astype(int)

            # 降水强度分类
            self.df['rain_intensity'] = pd.cut(self.df['prcp'],
                                               bins=[-np.inf, 0, 10, 25, 50, np.inf],
                                               labels=['无降水', '小雨', '中雨', '大雨', '暴雨'])

            # 连续降水天数
            self.df['rain_days'] = (self.df['prcp'] > 0).astype(int).rolling(window=7).sum()

            # 极端降水的持续性
            self.df['extreme_rain_days'] = self.df['is_extreme_rain'].rolling(window=7).sum()

    def create_weather_pattern_features(self):
        """创建天气模式特征"""
        if all(f in self.df.columns for f in ['T', 'U', 'P']):
            # 温湿指数 (体感温度指标)
            self.df['temp_humidity_index'] = 0.8 * self.df['T'] + (self.df['U'] / 100) * (self.df['T'] - 14.4) + 46.4

            # 大气稳定性指标
            self.df['pressure_trend'] = self.df['P'].diff()
            self.df['temp_pressure_index'] = self.df['T'] * self.df['P']

            # 湿度变化率
            self.df['humidity_change'] = self.df['U'].diff()

            # 天气变化模式
            self.df['weather_change_pattern'] = (self.df['pressure_trend'].apply(lambda x: 1 if x > 0 else -1) +
                                                 self.df['humidity_change'].apply(lambda x: 1 if x > 0 else -1))

    def create_attention_features(self):
        """创建适合注意力机制的特征"""
        # 特征重要性权重（基于与降水量的相关性）
        if 'prcp' in self.df.columns:
            correlations = {}
            for feature in self.key_features:
                if feature != 'prcp':
                    correlations[feature] = abs(self.df[feature].corr(self.df['prcp']))

            # 根据相关性创建加权特征
            for feature in correlations:
                self.df[f'{feature}_weighted'] = self.df[feature] * correlations[feature]

        # 创建特征组合
        if 'T' in self.df.columns and 'U' in self.df.columns:
            self.df['temp_humidity_comb'] = self.df['T'] * self.df['U']
        if 'P' in self.df.columns and 'ff10' in self.df.columns:
            self.df['pressure_wind_comb'] = self.df['P'] * self.df['ff10']

    def create_all_features(self):
        """创建所有特征"""
        print("1. 正在预处理特征...")
        self.preprocess_features()

        print("2. 正在创建时间特征...")
        self.create_time_features()

        print("3. 正在创建序列特征...")
        self.create_sequence_features()

        print("4. 正在创建极端降水特征...")
        self.create_extreme_rainfall_features()

        print("5. 正在创建天气模式特征...")
        self.create_weather_pattern_features()

        print("6. 正在创建注意力机制特征...")
        self.create_attention_features()

        # 删除包含缺失值的行
        self.df = self.df.dropna()

        # 打印特征信息
        print("\n特征工程总结：")
        print(f"特征总数：{len(self.df.columns)}")
        print("\n特征类别：")
        print("- 时间特征")
        print("- 序列特征（用于LSTM）")
        print("- 极端降水特征")
        print("- 天气模式特征")
        print("- 注意力机制特征")

        return self.df

    def prepare_lstm_sequences(self, sequence_length=14):
        """准备LSTM模型的输入序列"""
        # 选择用于LSTM的特征
        lstm_features = (self.key_features +
                         [col for col in self.df.columns if 'rolling' in col or 'trend' in col])

        # 创建序列数据
        sequences = []
        labels = []

        for i in range(len(self.df) - sequence_length):
            sequences.append(self.df[lstm_features].iloc[i:i + sequence_length].values)
            labels.append(self.df['prcp'].iloc[i + sequence_length])

        return np.array(sequences), np.array(labels)


if __name__ == "__main__":
    # 读取数据
    print("正在读取数据...")
    df = pd.read_csv('data_imputed.csv')
    df['date'] = pd.to_datetime(df['date'])

    # 创建特征工程实例
    engineer = RainfallFeatureEngineer(df)

    # 创建特征
    print("\n开始特征工程处理...")
    df_featured = engineer.create_all_features()

    # 准备LSTM序列数据
    print("\n正在准备LSTM序列数据...")
    sequences, labels = engineer.prepare_lstm_sequences()

    # 保存处理后的数据
    print("\n正在保存处理后的数据...")
    df_featured.to_csv('data_featured.csv', index=False)
    np.save('lstm_sequences.npy', sequences)
    np.save('lstm_labels.npy', labels)

    print("\n特征工程处理完成！")
    print(f"数据形状：{df_featured.shape}")
    print(f"LSTM序列数据形状：{sequences.shape}")
    print(f"标签数据形状：{labels.shape}")