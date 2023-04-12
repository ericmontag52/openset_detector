import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

def plot_ex_dataset(dataset, img_file, num_sample=5, img_size=28):
    #num_sample = 5

    sample_index = np.random.choice(dataset.shape[0], num_sample)

    f, ax = plt.subplots(1, num_sample)

    for i in range(num_sample):
        img = resize(dataset[sample_index[i]], (img_size, img_size))
        ax[i].imshow(img, cmap='gray')
        ax[i].axis('off')

    plt.savefig(img_file)
    
    
def making_batch(data_size, sample_size, data_x, data_y, img_size=28, num_label=10):
    # Making batches(testing)
    batch_idx = np.random.choice(data_size, sample_size)

    batch_x = np.zeros([sample_size, img_size, img_size, 1])
    batch_y = np.zeros([sample_size, num_label])

    for i in range(batch_idx.shape[0]):
        batch_x[i,:,:,:] = resize(data_x[batch_idx[i], :], (img_size, img_size, 1))
        batch_y[i,:]     = data_y[batch_idx[i], :]

    return batch_x, batch_y
    

def get_accuracy(prediction, label):
    count_correct = 0
    for j in range(prediction.shape[0]):
        if np.argmax(label[j, :]) == np.argmax(prediction[j, :]):
            count_correct += 1.0

    acc = count_correct / label.shape[0]
    return acc
