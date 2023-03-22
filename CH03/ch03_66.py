import pandas as pd

data = {
    "name": ["Jacky", "Peter", "John", "Joe"],
    "English": [90, 73, 78, 89],
    "chinese": [67, 70, 55, 45]
}
studentDf = pd.DataFrame(data)
# 刪除English欄位資料
new_df1 = studentDf.drop(["English"], axis=1)
print(new_df1)
# 刪除第一筆及第三筆人物資料
new_df2 = studentDf.drop([0,2], axis=0)
print(new_df2)
