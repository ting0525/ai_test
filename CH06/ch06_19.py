from numpy import expand_dims
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

img = load_img('dog.jpeg')  # 讀檔
img = img_to_array(img)  # 轉換為 numpy 陣列
img = expand_dims(img, 0)  # 擴充資料維度
# 建立生成器
datagen = ImageDataGenerator(brightness_range=[0.5,1.5])
# 準備迭代器
it = datagen.flow(img, batch_size=1)
fig = plt.figure()# 生成圖片並畫圖
for i in range(9):
    plt.subplot(3,3,1 + i)
    # 生成一個批次圖片
    batch = it.next()
    # 浮點型態轉化為整數型態才可以顯示
    image = batch[0].astype('uint32')
    fig.tight_layout()
    plt.imshow(image)
plt.show()
