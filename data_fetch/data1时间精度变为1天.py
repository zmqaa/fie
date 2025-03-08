import pandas as pd

# 假设数据已经加载到dataframe中
data = pd.read_csv('data1.csv')

# 将时间列转换为datetime格式
data['date'] = pd.to_datetime(data['date'], errors='coerce')

# 查看数据的头部
print(data.head())

# 提取日期部分并创建新的日期列
data['date'] = data['date'].dt.date

# 按日期进行聚合，计算每天的平均值
daily_data = data.groupby('date').agg({
    'T': 'mean',
    'Po': 'mean',
    'P': 'mean',
    'Pa': 'mean',
    'U': 'mean',
    'ff10': 'mean'
}).reset_index()

print(daily_data)

daily_data.to_csv('data1_1.csv', index=False)

df = pd.read