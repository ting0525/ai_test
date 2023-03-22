import pandas as pd

data = {
    "name": ["Jacky", "Peter", "John", "Joe"],
    "English": [90, 73, 78, 89],
    "chinese": [67, 70, 55, 45]
}
StudentDF = pd.DataFrame(data)
# 自訂索引值
StudentDF.columns = ["student_name", "English_score", "chinese_score"]
StudentDF.index = ["NO_1", "N0_2", "NO_3","N0_4"]  # 自訂欄位名稱
print(StudentDF)