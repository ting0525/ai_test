import pandas as pd
# 建立Pandas Series物件
data1 = pd.Series(["Mon","Tue","Wed","Thu"])
data2 = pd.Series(["Fri","Sat","Sun"])
data = data1.append(data2,ignore_index=True)
print(data)
