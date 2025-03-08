from data_processor import DataProcessor
import time


def main():
    # 设置输入输出路径
    input_file = 'data.csv'
    output_dir = 'processed_data'

    # 创建数据处理器
    processor = DataProcessor(input_file, output_dir)

    # 记录开始时间
    start_time = time.time()

    try:
        # 执行数据处理
        df_featured, sequences, labels = processor.process_data()

        # 计算处理时间
        processing_time = time.time() - start_time
        processor.log_message(f"\n总处理时间: {processing_time:.2f} 秒")

    except Exception as e:
        processor.log_message(f"\n处理过程中出现错误: {str(e)}")
        raise


if __name__ == "__main__":
    main()