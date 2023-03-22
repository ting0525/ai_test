import pandas as pd

data = {
    "name": ["Jacky", "Peter", "John", "Joe"],
    "English": [90, 73, 78, 89],
    "chinese": [67, 70, 55, 45]
}
studentDf = pd.DataFrame(data)
# 修改索引值為0的English欄位資料
studentDf.at[0, "English"] = 70
# 修改索引值為2的第三個欄位資料
studentDf.iat[2, 2] = 100
print(studentDf)
