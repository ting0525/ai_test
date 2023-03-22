import pandas as pd
s1 = pd.Series(['a', 'b', 'c'],
              index=[1, 3, 5])
print(s1)
s2 = s1.reindex(range(6), fill_value=0)
print(s2)

