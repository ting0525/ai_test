import pandas as pd

s = pd.Series([3, 4, 1, -4], index=['Mon', 'Tue', 'Wed', 'Thu'])
s1 = s[s>0]
print(s1)

