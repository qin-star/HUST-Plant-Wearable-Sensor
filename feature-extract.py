
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_excel(io=r"C:\Users\jiang\Desktop\all_data\20231227支撑材料\原始数据\无噪声\20231221数据增强后.xlsx")

#print(df)
#print(type(df))
print(df.values)
print(df.values[0,9])    #通过df.valued[]的方式引用数据
print(type(df.values[3,9]))
row = len(df.values)
print(row)  # 输出行数，（数据的组数）

# 取出前k列的数据：
k = 9
data = np.ones((row, k), dtype = float)
for i in range(k):                  #  后续默认i表示列数，j表示行数
    for j in range(row):
        data[j,i] = df.values[j,i]

# print(data.shape)
# print(data)


#  特征选择
# 1、channel5的值。
data_feature1 = np.zeros((row, 1))  # 用来储存feature
for j in range(row):
    data_feature1[j] = data[j,4]
print(data_feature1)

# 2、峰值前与x轴包裹的面积，采用梯形公式，设步长为1。峰值是第5个，所以总共4个梯形
dx = 1
data_feature2 = np.zeros((row, 1))  # 用来储存feature
for j in range(row):
    s = 0
    for i in range(4):
        s += (data[j, i] + data[j, i+1])*dx/2
    data_feature2[j] = s
print(data_feature2)

# 3、峰值后与x轴包裹的面积，采用梯形公式，设步长为1。峰值是第5个，所以总共3个梯形
dx = 1
data_feature3 = np.zeros((row, 1))  # 用来储存feature
for j in range(row):
    s = 0
    for i in range(4,7):
        s += (data[j, i] + data[j, i+1])*dx/2
    data_feature3[j] = s
print(data_feature3)

# 4、第3到第5个点的斜率
dx = 1
data_feature4 = np.zeros((row, 1))  # 用来储存feature
for j in range(row):
    data_feature4[j] = (data[j,4] - data[j,2])/(2*dx)
print(data_feature4)

# 5、第5到第8个点的斜率,
dx = 1
data_feature5 = np.zeros((row, 1))  # 用来储存feature
for j in range(row):
    data_feature5[j] = (data[j,7] - data[j,4])/(3*dx)
print(data_feature5)

# 6、标准差
data_feature6 = np.zeros((row, 1))
for j in range(row):
    data_feature6[j] = np.std(data[j,0:8])
print(data_feature6)

# 7、峰值前后面积比
data_feature7 = np.zeros((row, 1))
data_feature7 = data_feature2/data_feature3
print(data_feature7)

# 8、平均值
data_feature8 = np.zeros((row, 1))
for j in range(row):
    data_feature8[j] = np.mean(data[j,0:8])
print(data_feature8)

# 9、中值
data_feature9 = np.zeros((row, 1))
for j in range(row):
    data_feature9[j] = np.median(data[j,0:8])
print(data_feature9)

# 10、峰值前后斜率比
data_feature10 = np.zeros((row, 1))
data_feature10 = data_feature4/data_feature5
print(data_feature10)

# 11.NIR
data_feature11 = np.zeros((row, 1))
for j in range(row):
    data_feature11[j] = data[j, 8]
print(data_feature11)

# 12、第5到第7个点的斜率绝对值
dx = 1
data_feature12 = np.zeros((row, 1))  # 用来储存feature
for j in range(row):
    data_feature12[j] = abs((data[j,6] - data[j,4])/(2*dx))
print(data_feature12)

# 13、第7到第8个点的斜率绝对值,
dx = 1
data_feature13 = np.zeros((row, 1))  # 用来储存feature
for j in range(row):
    data_feature13[j] = abs((data[j,7] - data[j,6])/dx)
print(data_feature13)

# 14、
data_feature14 = np.zeros((row, 1))
data_feature14 = data_feature12/data_feature13
print(data_feature14)

# 15.chan7-chan5
data_feature15 = np.zeros((row, 1))  # 用来储存feature
for j in range(row):
    data_feature15[j] = data[j,6]-data[j,4]
print(data_feature15)

# 16. chan6 - 0.5*(chan5+chan8)
data_feature16 = np.zeros((row, 1))  # 用来储存feature
for j in range(row):
    data_feature16[j] = data[j,5]-0.5*(data[j,4]+data[j,7])
print(data_feature16)

# 17.(chan5-chan6)/(chan6-chan8)
data_feature17 = np.zeros((row, 1))  # 用来储存feature
for j in range(row):
    data_feature17[j] = (data[j,4]-data[j,5])/(data[j,5]-data[j,7])
print(data_feature17)

# 18.chan5 - chan8
data_feature18 = np.zeros((row, 1))  # 用来储存feature
for j in range(row):
    data_feature18[j] = data[j,4]-data[j,7]
print(data_feature18)

#  将需要的特征写入excel
def pd_toexcel(data,filename): # pandas库储存数据到excel
    dfData = { # 用字典设置DataFrame所需数据
    'chan5':data[0][0],
    '标准差': data[1][0],
    '(chan5-chan6)/(chan6-chan8)':data[2][0],
    '中值':data[3][0],                      #'峰值前后的斜率比':data[3][0],
    'chan7-chan5': data[4][0],
    'chan6 - 0.5*(chan5+chan8)': data[5][0],
    'chan5 - chan8': data[6][0],
    }
    df = pd.DataFrame(dfData) # 创建DataFrame
    df.to_excel(filename,index=False) # 存表，去除原始索引列（0,1,2...）
data_feature_all = [data_feature1.T, data_feature6.T, data_feature17.T, data_feature9.T, data_feature15.T, data_feature16.T, data_feature18.T]  # 注意这里转置了一下
#print(data_feature_all)
#print(len(data_feature_all))
#print(data_feature_all[3][0])
pd_toexcel(data_feature_all, r"C:\Users\jiang\Desktop\all_data\20231227支撑材料\原始数据\无噪声\fea_20231221数据增强.xlsx")





