import matplotlib.pyplot as plt
import matplotlib.image as mpimg   # mpimg 用於讀取圖片
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

gen_path = r'flower'       # 存放類別的資料夾
savePath = 'train_image'   # 產生的圖片存放路徑
def print_result(path):
    imagelist = os.listdir(path)
    fig = plt.figure()
    for i in range(len(imagelist)):
        imgpath = path + '\\' + imagelist[i]
        img = mpimg.imread(imgpath)
        sub_img = fig.add_subplot(441 + i)
        sub_img.imshow(img)
        fig.tight_layout()
    plt.show()
    return fig
# 產生一個 ImageDataGenerator 類別物件
image_gen = ImageDataGenerator(rescale=1,rotation_range=90)
# 建立迭代器
# 指定讀取數據路徑(directory)與設定生成的影像存取路徑(save_to_dir)
# target_size 可以指定經過處理後的圖形大小多少
it = image_gen.flow_from_directory(directory='flower',
                                   batch_size=5,
                                   save_to_dir = savePath,
                                   target_size=(128,128),
                                   classes=['rose'],
                                   save_prefix='trans_',
                                   save_format='jpg')

# 利用迴圈來做迭代產生影像 (本範例中只產生一次迭代)
for data_batch,_ in it:
    print(data_batch.shape)
    break
# 印出產生的資料
fig = print_result(savePath)