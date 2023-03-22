from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model

os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'
# 載入影像後做一個尺度大小設定
img = image.load_img("cat001.jpg",target_size=(224,224))
img = np.asarray(img)
plt.imshow(img)
img = np.expand_dims(img, axis=0)
plt.show(block=True)
# 載入模型
saved_model = load_model("vgg16.h5")
# 模型預測
output = saved_model.predict(img)
if output[0][0] > output[0][1]:
    print("cat")
else:
    print('dog')

