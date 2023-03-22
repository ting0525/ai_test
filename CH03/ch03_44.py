import pandas as pd
# 建立Pandas Series物件
data = pd.Series(["Mon","Tue","Wed","Thu",
                  "Fri","Sat","Sun",])
print(data)

booldata = [True,False,False,True]
booldataS = pd.Series(booldata)
print(booldataS)
IntData = [10,20,30,40]
IntDataS = pd.Series(IntData)
print(IntDataS)

