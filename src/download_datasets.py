import keras.datasets
import numpy as np

validation_ratio = .1
num_label = 10 # 0 ~ 9

def mnist(validation_ratio):
    (train_img, train_l), (testval_img, testval_l) = keras.datasets.mnist.load_data()
    test_len = testval_l.shape[0]
    validation_len = int(test_len * validation_ratio)

    train_x = train_img
    test_x = testval_img[validation_len : test_len, :]
    validation_x = testval_img[ : validation_len, :]

    train_y_index = train_l
    test_y_index = testval_l[validation_len : test_len]
    validation_y_index = testval_l[ : validation_len]

    train_y = np.zeros([train_y_index.shape[0], num_label])
    test_y = np.zeros([test_y_index.shape[0], num_label])
    validation_y = np.zeros([validation_y_index.shape[0], num_label])

    for i in range(train_y.shape[0]):
        train_y[i, train_y_index[i]] = 1

    for i in range(test_y.shape[0]):
        test_y[i, test_y_index[i]] = 1

    for i in range(validation_y.shape[0]):
        validation_y[i, validation_y_index[i]] = 1
        return (train_x, train_y), (validation_x,validation_y),(test_x,test_y)
    
def fashion_mnist(validation_ratio):
    (train_img, train_l), (testval_img, testval_l) = keras.datasets.fashion_mnist.load_data()
    test_len = testval_l.shape[0]
    validation_len = int(test_len * validation_ratio)

    train_x = train_img
    test_x = testval_img[validation_len : test_len, :]
    validation_x = testval_img[ : validation_len, :]

    train_y_index = train_l
    test_y_index = testval_l[validation_len : test_len]
    validation_y_index = testval_l[ : validation_len]

    train_y = np.zeros([train_y_index.shape[0], num_label])
    test_y = np.zeros([test_y_index.shape[0], num_label])
    validation_y = np.zeros([validation_y_index.shape[0], num_label])

    for i in range(train_y.shape[0]):
        train_y[i, train_y_index[i]] = 1

    for i in range(test_y.shape[0]):
        test_y[i, test_y_index[i]] = 1

    for i in range(validation_y.shape[0]):
        validation_y[i, validation_y_index[i]] = 1
        return (train_x, train_y), (validation_x,validation_y),(test_x,test_y)
