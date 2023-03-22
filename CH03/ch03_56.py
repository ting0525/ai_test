import pandas as pd
# 利用二維列表創建 DataFrame
Peopledata = [['Jacky',35],['Peter',40],['Joe',25]]
df = pd.DataFrame(Peopledata,columns=['Name','Age'])
print(df)
