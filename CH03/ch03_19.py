import numpy as np

a = np.array([1.2, 3.456, 125.333, 0.657, 25.567])
print  ('原來的數：')
print (a)

print ('數字捨入後：')
print (np.around(a))
print (np.around(a, decimals = 2))
print (np.around(a, decimals = -1))

