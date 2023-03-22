import tensorflow as tf

(train_image,train_label),(test_image,test_label)=\
    tf.keras.datasets.fashion_mnist.load_data()

print("train_image.shape :",train_image.shape)
print("train_image.shape :",train_label.shape)
train_data, valid_data, test_data = tf.split(train_image,
                            [36000,12000,12000],axis=0)
print("train_data.shape :", train_data.shape)
print("valid_data.shape :", valid_data.shape)
print("test_data.shape :", test_data.shape)