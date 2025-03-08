import pandas as pd

data1 = pd.read_csv('data1_1.csv')
data2 = pd.read_csv('data2.csv')

# 将日期列转换为 datetime 格式，假设日期列的名字为 'date'
data1['date'] = pd.to_datetime(data1['date'], errors='coerce')
data2['date'] = pd.to_datetime(data2['time'], errors='coerce')

# 删除掉包含无效日期的行
data1.dropna(subset=['date'], inplace=True)
data2.dropna(subset=['date'], inplace=True)

# 按日期列进行合并，合并方式为内连接（只保留两个数据集中都有的日期）
merged_data = pd.merge(data1, data2, on='date', how='inner')

merged_data = merged_data.drop(columns=['time'])

print(merged_data)
merged_data.to_csv('data.csv', index=False)