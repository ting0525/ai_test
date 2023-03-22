import pandas as pd
numbers = pd.Series([10, 20, 30, 40, 50, 60])
print(numbers.max())   # 找最大值
print(numbers.nlargest(3))  # 最大的3個數值
print(numbers.min())   # 找最小值
print(numbers.nsmallest(3))  # 最小的3個數值
print(numbers.sum())  # 計算總和
print(numbers.mean())  # 計算平均值
