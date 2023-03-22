import pandas as pd
# 建立Pandas Series物件
data = pd.Series(["Mon","Tue","Wed","Thu",
                  "Fri","Sat","Sun",],
                 index = ["m1","m2","m3","m4",
                          "m5","m6","m7"])
print("data[\"m1\"]:",data["m1"]) # 列印 index = m1 的資料
print("data[0]:",data[0]) # 列印 index = 0 的資料

a = 'm1' in data
print(a)




