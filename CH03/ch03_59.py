import pandas as pd

data = {
    "name": ["Jacky", "Peter", "John", "Joe"],
    "English": [90, 73, 78, 89],
    "chinese": [67, 70, 55, 45]
}
StudentDF = pd.DataFrame(data)
# 取得全部欄位資料
print(StudentDF)
# 取得單一欄位資料
print(StudentDF["name"])
# 取得多個欄位資料
print(StudentDF[["name","English"]])
# 取得多個索引的欄位資料
print(StudentDF[0:4])