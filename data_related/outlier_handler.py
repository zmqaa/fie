import pandas as pd
import numpy as np
from scipy import stats


class WeatherOutlierHandler:
    def __init__(self, df):
        self.df = df
        # 将特征分为降水特征和其他气象特征
        self.rainfall_cols = ['prcp']
        self.other_weather_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                                   if col not in self.rainfall_cols and col != 'date']

    def handle_rainfall_outliers(self, rainfall_col='prcp'):
        """专门处理降水数据的异常值"""
        print(f"\n正在处理降水数据异常值...")
        df_rainfall = self.df[rainfall_col].copy()

        # 1. 基本检查
        # 处理负值
        invalid_rain = (df_rainfall < 0)
        if invalid_rain.any():
            print(f"发现{invalid_rain.sum()}个负降水值，已将其设置为0")
            df_rainfall[invalid_rain] = 0

        # 2. 按月份分别处理极端值
        monthly_outliers = pd.Series(dtype=bool)

        for month in range(1, 13):
            month_data = df_rainfall[self.df['date'].dt.month == month]
            if len(month_data) == 0:
                continue

            # 计算月降水量的四分位数
            Q1 = month_data.quantile(0.25)
            Q3 = month_data.quantile(0.75)
            IQR = Q3 - Q1

            # 使用更宽松的阈值判定极端降水
            upper_bound = Q3 + 3 * IQR  # 使用3倍IQR而不是通常的1.5倍

            # 标记当前月份的异常值
            month_mask = (self.df['date'].dt.month == month)
            monthly_outliers = monthly_outliers.reindex(self.df.index).fillna(False)
            monthly_outliers[month_mask] = (df_rainfall[month_mask] > upper_bound)

        # 3. 处理识别出的极端降水
        if monthly_outliers.any():
            print(f"\n发现{monthly_outliers.sum()}个潜在的极端降水值：")
            print("\n极端降水事件统计：")
            extreme_events = self.df[monthly_outliers][['date', rainfall_col]]
            print(extreme_events.sort_values(by=rainfall_col, ascending=False).head())

            # 不直接修改极端值，而是添加标记
            self.df['is_extreme_rainfall'] = monthly_outliers

            # 计算极端降水的一些统计特征
            self.df['rainfall_percentile'] = self.df.groupby(self.df['date'].dt.month)[rainfall_col].transform(
                lambda x: pd.qcut(x, q=100, labels=False, duplicates='drop'))

        return self.df

    def handle_other_weather_outliers(self):
        """处理其他气象要素的异常值"""
        print("\n正在处理其他气象要素的异常值...")

        for col in self.other_weather_cols:
            # 使用传统的IQR方法检测异常值
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = ((self.df[col] < lower_bound) |
                        (self.df[col] > upper_bound))

            if outliers.any():
                print(f"\n{col}列发现{outliers.sum()}个异常值")

                # 使用移动中位数替换异常值
                self.df.loc[outliers, col] = self.df[col].rolling(
                    window=5, center=True, min_periods=1
                ).median()[outliers]

    def add_rainfall_statistics(self):
        """添加降水统计特征"""
        print("\n正在添加降水统计特征...")

        # 按月统计降水特征
        monthly_stats = self.df.groupby(self.df['date'].dt.month)['prcp'].agg([
            ('monthly_rain_mean', 'mean'),
            ('monthly_rain_std', 'std'),
            ('monthly_rain_days', lambda x: (x > 0).sum())
        ])

        # 将统计量合并回原数据
        self.df = self.df.merge(
            monthly_stats,
            left_on=self.df['date'].dt.month,
            right_index=True,
            how='left'
        )

        # 添加连续降水/无降水天数
        self.df['continuous_rain_days'] = (
            (self.df['prcp'] > 0)
            .astype(int)
            .groupby((self.df['prcp'] == 0).astype(int).cumsum())
            .cumsum()
        )

        self.df['continuous_dry_days'] = (
            (self.df['prcp'] == 0)
            .astype(int)
            .groupby((self.df['prcp'] > 0).astype(int).cumsum())
            .cumsum()
        )

    def process_all(self):
        """执行所有处理步骤"""
        print("开始处理异常值...")

        # 1. 首先处理降水数据
        self.df = self.handle_rainfall_outliers()

        # 2. 处理其他气象要素
        self.handle_other_weather_outliers()

        # 3. 添加降水统计特征
        self.add_rainfall_statistics()

        print("\n异常值处理完成！")
        return self.df


if __name__ == "__main__":
    # 读取数据
    print("正在读取数据...")
    df = pd.read_csv('data_loaded.csv')
    df['date'] = pd.to_datetime(df['date'])

    # 创建异常值处理实例
    handler = WeatherOutlierHandler(df)

    # 处理异常值
    df_cleaned = handler.process_all()

    # 保存处理后的数据
    df_cleaned.to_csv('data_cleaned.csv', index=False)
    print("\n数据已保存至 'data_cleaned.csv'")

    # 输出数据处理总结
    print("\n数据处理总结：")
    print(f"数据形状：{df_cleaned.shape}")
    print("\n降水特征统计：")
    print(df_cleaned[['prcp', 'is_extreme_rainfall', 'rainfall_percentile']].describe())