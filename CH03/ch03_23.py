import numpy as np

a = np.array([2, 5, 10])
print('傳入的次方項是:')
print(a)
print('呼叫以2為底的 power 函數：')
print(np.power(2, a))
print('傳入的底數是:')
b = np.array([1, 2, 3])
print(b)
print('呼叫以傳入a為底,b為指數的 power 函數：')
print(np.power(a, b))

