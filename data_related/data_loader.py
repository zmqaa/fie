import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, file_path='data.csv', missing_threshold=0.4):
        """
        初始化数据加载器
        Args:
            file_path: 数据文件路径
            missing_threshold: 缺失值比例阈值，超过这个比例的列将被删除
        """
        self.file_path = file_path
        self.missing_threshold = missing_threshold

    def check_missing_values(self, df):
        """检查并处理缺失值"""
        # 计算每列的缺失值比例
        missing_ratio = df.isnull().sum() / len(df)

        # 找出缺失值比例超过阈值的列
        high_missing_cols = missing_ratio[missing_ratio > self.missing_threshold].index

        if len(high_missing_cols) > 0:
            print("\n发现缺失值比例过高的列：")
            for col in high_missing_cols:
                print(f"- {col}: {missing_ratio[col]:.2%} 缺失")

            # 删除这些列
            df = df.drop(columns=high_missing_cols)
            print(f"\n已删除 {len(high_missing_cols)} 个缺失值比例超过 {self.missing_threshold:.0%} 的列")

        # 显示剩余列的缺失值情况
        remaining_missing = df.isnull().sum()
        remaining_missing = remaining_missing[remaining_missing > 0]

        if len(remaining_missing) > 0:
            print("\n剩余列的缺失值情况：")
            for col, missing_count in remaining_missing.items():
                print(f"- {col}: {missing_count}个缺失值 ({missing_count / len(df):.2%})")

        return df

    def load_data(self):
        """加载数据并进行基础处理"""
        try:
            print(f"正在读取数据文件: {self.file_path}")
            df = pd.read_csv(self.file_path)

            # 基本信息
            print("\n原始数据基本信息：")
            print(f"- 数据形状: {df.shape}")
            print(f"- 列名: {', '.join(df.columns)}")

            # 转换日期列
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                print("\n已将'date'列转换为日期时间格式")

            # 处理缺失值过多的列
            df = self.check_missing_values(df)

            # 显示数据类型信息
            print("\n各列数据类型：")
            for col, dtype in df.dtypes.items():
                print(f"- {col}: {dtype}")

            # 显示数值列的基本统计信息
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                print("\n数值列统计信息：")
                print(df[numeric_cols].describe())

            print(f"\n数据加载完成，最终数据形状: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"数据加载过程中出错: {str(e)}")
            raise


if __name__ == "__main__":
    # 测试数据加载
    loader = DataLoader('data.csv')
    df = loader.load_data()

    # 保存处理后的数据
    df.to_csv('data_loaded.csv', index=False)
    print("\n数据已保存至 'data_loaded.csv'")