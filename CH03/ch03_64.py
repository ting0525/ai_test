import pandas as pd

data = {
    "name": ["Jacky", "Peter", "John", "Joe"],
    "English": [90, 73, 78, 89],
    "chinese": [67, 70, 55, 45]
}
studentDf = pd.DataFrame(data)
otherStudDf = pd.DataFrame({
    "name":["Steven"],
    "English":[90],
    "chinese":[80],
})
newStudDf = pd.concat([studentDf,otherStudDf],ignore_index=True)
print(newStudDf)