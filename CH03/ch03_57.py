import pandas as pd
# 利用字典創建 DataFrame
Peopledata = [{'Jacky':35,'Peter':40,'Joe':25},
              {'Jacky':'Phd','Joe':'master'}]
# 利用 index 參數來設定列索引
df = pd.DataFrame(Peopledata,index={'Name','Degree'})
print(df)