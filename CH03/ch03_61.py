import pandas as pd

data = {
    "name": ["Jacky", "Peter", "John", "Joe"],
    "English": [90, 73, 78, 89],
    "chinese": [67, 70, 55, 45]
}
studentdf = pd.DataFrame(data)
print(studentdf.loc[[1,3], ["name","English"]])
