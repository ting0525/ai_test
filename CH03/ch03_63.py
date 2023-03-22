import pandas as pd

data = {
    "name": ["Jacky", "Peter", "John", "Joe"],
    "English": [90, 73, 78, 89],
    "chinese": [67, 70, 55, 45]
}
studentdf = pd.DataFrame(data)
# 新增一個欄位資料
studentdf.insert(2, column="Math", value=[90, 75, 80, 90])
print(studentdf)
# 新增一筆資料
NewStudDf = studentdf.append({
    "name":"Steven",
    "English":90,
    "chinese":80,
    "Math":100},ignore_index=True)
print(NewStudDf)
