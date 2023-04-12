import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io
import csv
import random
import time

from skimage.transform import resize
import src.deep_ensembels as de
import src.download_datasets as dd
from src import utils

img_size=28
num_label=10
networks=[0,0,0,0,0]

class Test:
    def __init__(self, name='test', model='deep_ensembles_models/sess1', img_size=28, train_dataset='mnist', non_dataset='fashion', num_label=10, gpu_fraction=0.5, num_net=5):
        #Config stuff here
        self.name = name
        self.model = model
        self.img_size = img_size
        self.train_dataset = train_dataset
        self.non_dataset = non_dataset
        self.num_label = num_label #Will want to break up into train and non in the future. The train and non have to have the same number of classes
        self.validation_ratio = 0.1 #Should not need to adjust, we want most of the data going to test anyways.
        self.gpu_fraction = gpu_fraction
        self.get_num_net(num_net)

        #Grabbing data
        self.get_train_data()
        self.get_non_train_data()

        tf.reset_default_graph() #For tensorflow purposes
        
        #Pulling in the NN
        self.initialize_network()
        self.create_session()

        #Testing the train and non train datasets
        self.test_train()
        self.test_non()


    #Data Functions
    def get_train_data(self):
        '''
        Function that gets you the test data for the data that has been trained on.
        Currently hard coded for mnist
        '''
        if self.train_dataset == 'mnist':
            (_,_),(_,_),(self.train_x, self.train_y) = dd.mnist(self.validation_ratio)
        elif self.train_dataset == 'fashion':
            (_,_),(_,_),(self.train_x, self.train_y) = dd.fashion_mnist(self.validation_ratio)
        elif self.train_dataset == 'cifar10':
            (_,_),(_,_),(self.train_x, self.train_y) = dd.cifar10(self.validation_ratio)
        elif self.train_dataset == 'cifar100':
            (_,_),(_,_),(self.train_x, self.train_y) = dd.cifar100(self.validation_ratio)

    def get_non_train_data(self):
        '''
        Function tha get your test data for the open set
        Currently hard coded for fashion mnist
        '''
        if self.non_dataset == 'mnist':
            (_,_),(_,_),(self.non_x, self.non_y) = dd.mnist(self.validation_ratio)
        elif self.non_dataset == 'fashion':
            (_,_),(_,_),(self.non_x, self.non_y) = dd.fashion_mnist(self.validation_ratio)
        elif self.non_dataset == 'cifar10':
            (_,_),(_,_),(self.non_x, self.non_y) = dd.cifar10(self.validation_ratio)
        elif self.non_dataset == 'cifar100':
            (_,_),(_,_),(self.non_x, self.non_y) = dd.cifar100(self.validation_ratio)

    def get_num_net(self, num_net):
        self.networks = []
        for i in range(num_net):
            self.networks.append(f'network{i}')
    
    def initialize_network(self):
        '''
        Function to initialize the networks
        Currently hardcoded for deep ensembles
        '''
        self.x_list = []
        self.y_list = []
        self.output_list = []
        self.loss_list = []
        self.train_list = []
        self.train_var_list = []

        for i in range(len(self.networks)):
            x_image, y_label, output, loss, train_opt, train_vars = de.get_network(self.networks[i])

            self.x_list.append(x_image)
            self.y_list.append(y_label)
            self.output_list.append(output)
            self.loss_list.append(loss)
            self.train_list.append(train_opt)
            self.train_var_list.append(train_vars)
    
    def create_session(self):
        '''
        Function to create a tf session in order to test the NN
        '''
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = self.gpu_fraction

        self.sess = tf.InteractiveSession(config=config)
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.sess, self.model)

    def test_train(self):
        '''
        Function to run the dataset that the NN has been trained through the NN
        '''
        self.test_n_images(self.train_x, self.train_y, self.name+f'/train_{self.non_dataset}')

    def test_non(self):
        '''
        Function to run the dataset that has not been trained through the NN
        '''
        self.test_n_images(self.non_x, self.non_y, self.name+f'/non_train_{self.non_dataset}')

    def test_n_images(self, dataset_x, dataset_y, file_name):
        '''
        Running n images through the session
        Currently hard coded for Deep ensembles
        '''
        sample_index = np.random.choice(dataset_x.shape[0], 100)

        fig, ax = plt.subplots(1, 100, figsize=(12,24))

        output_sample     = []
        output_sample_one = []

        f = open(f'results/{file_name}_results.csv','w')
        writer = csv.writer(f)
        writer.writerow(['time(s)', 'dataset_index','class','single_class','single_prob', 'ensemble_class','ensemble_prob'])
        for i, index in enumerate(sample_index):
            img = resize(dataset_x[index], (self.img_size, self.img_size))

            ax[i].imshow(img, cmap='gray')
            ax[i].axis('off')
            ax[i].set_title(str(i+1) + 'th')

            output_test = np.zeros([1, self.num_label])

            for net_index in range(len(self.networks)):
                x_temp = np.reshape(dataset_x[index, :], (1, self.img_size, self.img_size, 1))
                y_temp = np.reshape(dataset_y[index, :], (1, self.num_label))

                #Begin timer
                start = time.time()
                loss_temp, prob_temp = self.sess.run([self.loss_list[net_index], self.output_list[net_index]],
                                         feed_dict = {self.x_list[net_index]: x_temp, self.y_list[net_index]: y_temp})
                #End timer
                end = time.time()
                t = end - start

                # Add test prediction for get final prediction
                output_test += prob_temp

            # Get final test prediction
            e_prob = output_test / len(self.networks)
            s_prob   = prob_temp

            s_class = np.argmax(s_prob)
            e_class = np.argmax(e_prob)
            s_acc = s_prob[0,s_class]
            e_acc = e_prob[0,e_class]
            writer.writerow([t, index, np.argmax(y_temp), s_class, s_acc, e_class, e_acc])

        plt.savefig(f'results/{file_name}_img.png')
        

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, default='test', help='Name of the folder the data is saved to inside of results')
    parser.add_argument("-m", "--model", type=str, default='deep_ensembles_models/sess1', help='Name of the model that will be imported for testing')
    parser.add_argument("-s", "--img-size", type=int, default=28, help='Number of pixels across the img')
    parser.add_argument("-l", "--num-label", type=int, default=10, help='Number of classes in the datasets')
    parser.add_argument("-g", "--gpu-fraction", type=float, default=0.5, help='Percentage of the GPU being utilized')
    parser.add_argument("-d", "--dataset", type=str, default='mnist', help='The name of the dataset to train')
    parser.add_argument("-nd", "--non-dataset", type=str, default='fashion', help='The name of the non dataset to train')
    parser.add_argument("-nn", "--num_net", type=int, default=5, help='Number of networks in the ensemble')
    args = parser.parse_args()
    os.makedirs(f'results/{args.name}', exist_ok=True)
    Test(name=args.name, model=args.model, img_size=args.img_size, num_label=args.num_label, gpu_fraction=args.gpu_fraction, non_dataset=args.non_dataset, num_net=args.num_net)
