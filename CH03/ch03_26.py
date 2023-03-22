import numpy as np

A = np.array([
    [1,2],
    [3,3]])
B = np.array([
    [4,2],
    [2,3]])
# 矩陣相乘
print("A.dot(B) = ", A.dot(B))
print("A@B = ", A@B)
print("np.matmul(A,B) = ",np.matmul(A,B))

print("np.multiply(A,B) = ",np.multiply(A,B))
# 當A不可逆的時候, 求A的反矩陣
print("np.linalg.inv(A) = ",np.linalg.inv(A))
# 求A的轉置矩陣
print("A.T = ",A.T)
# 判斷A, B兩個矩陣是否相等
print("np.array_equal(A,B) = ",np.array_equal(A,B))

