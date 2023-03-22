import pandas as pd
# 利用 ndarrays 創建 DataFrame
Peopledata = {'Name':['Jacky', 'Peter', 'Joe'], 'Age':[35, 40, 25]}
df = pd.DataFrame(Peopledata)
print (df)
