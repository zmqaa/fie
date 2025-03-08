# Import Meteostat library and dependencies
from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Point, Daily

# Set time period
# 使用尽可能早的开始时间，直到现在
start = datetime(1950, 1, 1)  # Meteostat 通常从1950年代开始有数据
end = datetime.now()

# Create Point for Guangzhou (latitude, longitude, altitude)
guangzhou = Point(23.1291, 113.2644, 21)

# Get daily data
data = Daily(guangzhou, start, end)
data = data.fetch()

# Print basic statistics
print("数据概览：")
print(data.describe())
print("\n可用数据时间范围：")
print(f"开始时间：{data.index.min()}")
print(f"结束时间：{data.index.max()}")


# 保存数据到CSV文件（可选）
data.to_csv('data1.csv')