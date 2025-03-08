import pandas as pd
import numpy as np
import os
import time
from datetime import datetime

from data_loader import DataLoader
from outlier_handler import WeatherOutlierHandler
from gan import GANImputer
from feature_engineer import RainfallFeatureEngineer


class DataProcessor:
    def __init__(self, input_file='data.csv', output_dir='processed_data'):
        """
        初始化数据处理器
        Args:
            input_file: 输入数据文件路径
            output_dir: 输出数据保存目录
        """
        self.input_file = input_file
        self.output_dir = output_dir

        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 初始化日志文件
        self.log_file = os.path.join(output_dir, 'processing_log.txt')

    def log_message(self, message):
        """记录处理日志"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)

        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')

    def save_intermediate_data(self, data, stage_name):
        """保存中间处理结果"""
        filename = os.path.join(self.output_dir, f'data_{stage_name}.csv')
        data.to_csv(filename, index=False)
        self.log_message(f"数据已保存至: {filename}")

    def process_data(self):
        """执行完整的数据处理流程"""
        try:
            # 1. 数据加载
            self.log_message("开始数据加载...")
            loader = DataLoader(self.input_file)
            df = loader.load_data()
            self.save_intermediate_data(df, 'loaded')

            # 2. 异常值处理
            self.log_message("\n开始异常值处理...")
            outlier_handler = WeatherOutlierHandler(df)
            df_cleaned = outlier_handler.process_all()
            self.save_intermediate_data(df_cleaned, 'cleaned')

            # 3. GAN填充缺失值
            self.log_message("\n开始GAN缺失值填充...")
            imputer = GANImputer(df_cleaned)
            df_imputed = imputer.impute_missing_values()
            self.save_intermediate_data(df_imputed, 'imputed')

            # 4. 特征工程
            self.log_message("\n开始特征工程...")
            engineer = RainfallFeatureEngineer(df_imputed)
            df_featured = engineer.create_all_features()

            # 5. 准备深度学习数据
            self.log_message("\n准备深度学习数据序列...")
            sequences, labels = engineer.prepare_lstm_sequences()

            # 保存最终处理结果
            self.log_message("\n保存最终处理结果...")
            final_data_path = os.path.join(self.output_dir, 'final_data.csv')
            df_featured.to_csv(final_data_path, index=False)

            sequence_path = os.path.join(self.output_dir, 'lstm_sequences.npy')
            labels_path = os.path.join(self.output_dir, 'lstm_labels.npy')
            np.save(sequence_path, sequences)
            np.save(labels_path, labels)

            # 生成处理报告
            self.generate_report(df_featured, sequences, labels)

            self.log_message("数据处理完成！")
            return df_featured, sequences, labels

        except Exception as e:
            self.log_message(f"错误: {str(e)}")
            raise

    def generate_report(self, df_featured, sequences, labels):
        """生成数据处理报告"""
        report_path = os.path.join(self.output_dir, 'processing_report.txt')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== 数据处理报告 ===\n\n")

            # 1. 基本信息
            f.write("1. 数据基本信息\n")
            f.write("-----------------\n")
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"原始数据文件: {self.input_file}\n")
            f.write(f"最终数据形状: {df_featured.shape}\n")
            f.write(f"LSTM序列形状: {sequences.shape}\n")
            f.write(f"标签数据形状: {labels.shape}\n\n")

            # 2. 特征信息
            f.write("2. 特征信息\n")
            f.write("-----------------\n")
            f.write("数值型特征:\n")
            numeric_cols = df_featured.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                f.write(f"- {col}\n")
            f.write("\n")

            # 3. 降水统计
            f.write("3. 降水数据统计\n")
            f.write("-----------------\n")
            if 'prcp' in df_featured.columns:
                rain_stats = df_featured['prcp'].describe()
                f.write("降水量统计:\n")
                f.write(str(rain_stats) + "\n\n")

                if 'is_extreme_rainfall' in df_featured.columns:
                    extreme_count = df_featured['is_extreme_rainfall'].sum()
                    f.write(f"极端降水事件数量: {extreme_count}\n")

            # 4. 数据质量信息
            f.write("\n4. 数据质量信息\n")
            f.write("-----------------\n")
            missing_data = df_featured.isnull().sum()
            if missing_data.any():
                f.write("存在缺失值的特征:\n")
                f.write(str(missing_data[missing_data > 0]) + "\n")
            else:
                f.write("所有特征均无缺失值\n")

        self.log_message(f"处理报告已生成: {report_path}")