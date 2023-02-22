import keras.datasets

def mnist():
    (train_img, train_l), (testval_img, testval_l) = keras.datasets.mnist.load_data()
    return (train_img, train_l), (testval_img, testval_l)
    
def fashion_mnist():
    (_,_), (fashion_mnist_img, fashion_mnist_l) = keras.datasets.fashion_mnist.load_data()
    return (fashion_mnist_img, fashion_mnist_l)
