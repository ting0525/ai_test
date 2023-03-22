import pandas as pd
s1 = pd.Series([1, 2, 3, 4],
              index=['d', 'b', 'a', 'c'])
print(s1)
s2 = s1.reindex(['a', 'b', 'c', 'd', 'e'])
print(s2)

