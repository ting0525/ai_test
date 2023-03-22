import pandas as pd
import numpy as np

data = {
    "name": ["Jacky", "Peter", "John", np.NaN],
    "English": [90, np.NaN, 70, 80],
    "math": [88, 65, 67, 95],
    "chinese": [81, 92, 73, 77]
}
studentDf = pd.DataFrame(data)
# 印出原始 DF
print(studentDf)
newstudentDf = studentDf.dropna()
# 印出刪除 NaN 欄位後 DF
print(newstudentDf)
