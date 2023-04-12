import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io
import cv2
import random
import argparse

import src.deep_ensembels as de
import src.download_datasets as dd
from src import utils

class Train():
    def __init__(self, name='test', epoch = 5000, batch_size = 256, model='deep_ensembles_models/sess1', img_size=28, dataset='mnist', num_label=10, gpu_fraction=1.0, validation_ratio=0.1, num_net=5):
        #CONFIG STUFF
        self.name = name
        self.epoch = epoch
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_flat_size = self.img_size * self.img_size
        self.dataset = dataset
        self.num_label = num_label #DO WE WANT TO MOVE THIS TO THE DOWNLOAD DATASET?
        self.model = model
        self.validation_ratio = validation_ratio
        self.gpu_fraction = gpu_fraction
        self.get_num_net(num_net)
        
        self.get_data()
    
        utils.plot_ex_dataset(self.train_x, f'results/{name}/train_data_ex.png', num_sample=5, img_size=self.img_size) #Plotting Datasets examples

        tf.reset_default_graph() #For tensorflow purposes

        self.initialize_network()
        self.create_session()  
        
        self.train()

        self.saver.save(self.sess, self.model)
        
    def get_data(self):
        '''
        Function to bring in the data for the training session
        Currently hard coded for MNIST
        '''
        if self.dataset == 'mnist':
            (self.train_x, self.train_y), (self.validation_x, self.validation_y),(self.test_x, self.test_y) = dd.mnist(self.validation_ratio)
        elif self.dataset =='fashion':
            (self.train_x, self.train_y), (self.validation_x, self.validation_y),(self.test_x, self.test_y) = dd.fashion_mnist(self.validation_ratio)
        elif self.dataset == 'cifar10':
            (self.train_x, self.train_y), (self.validation_x, self.validation_y),(self.test_x, self.test_y) = dd.cifar10(self.validation_ratio)
        elif self.dataset == 'cifar100':
            (self.train_x, self.train_y), (self.validation_x, self.validation_y),(self.test_x, self.test_y) = dd.cifar100(self.validation_ratio)
        print(f"\nPrinting {self.dataset} dataset shape")
        print("\nTraining X shape: " + str(self.train_x.shape))
        print("Testing X shape: " + str(self.test_x.shape))
        print("Validation X shape: " + str(self.validation_x.shape))

        print("\nTraining Y shape: " + str(self.train_y.shape))
        print("Testing Y shape: " + str(self.test_y.shape))
        print("Validation Y shape: " + str(self.validation_y.shape) + '\n')
        
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
        self.saver = tf.train.Saver()
        
    def train(self):
        # Initialize data for printing
        loss_check     = np.zeros(len(self.networks))
        acc_check      = np.zeros(len(self.networks))
        acc_check_test = np.zeros(len(self.networks))
        acc_check_test_final = 0

        # Set parameters for printing and testing
        num_print = 100
        test_size = 10

        train_data_num = self.train_x.shape[0]
        test_data_num  = self.test_x.shape[0]

        for iter in range(self.epoch):
            output_temp = np.zeros([test_size, self.num_label])

            # Making batches(testing)
            batch_x_test, batch_y_test = utils.making_batch(test_data_num, test_size, self.test_x, self.test_y)

            for i in range(len(self.networks)):
                # Making batches(training)
                batch_x, batch_y = utils.making_batch(train_data_num, self.batch_size, self.train_x, self.train_y)

                # Training
                _, loss, prob = self.sess.run([self.train_list[i], self.loss_list[i], self.output_list[i]],
                                 feed_dict = {self.x_list[i]: batch_x, self.y_list[i]: batch_y})

                # Testing
                loss_test, prob_test = self.sess.run([self.loss_list[i], self.output_list[i]],
                                         feed_dict = {self.x_list[i]: batch_x_test, self.y_list[i]: batch_y_test})

                # Add test prediction for get final prediction
                output_temp += prob_test

                # Calculate Accuracy (Training)
                acc_training = utils.get_accuracy(prob, batch_y)

                # Calculate Accuracy (testing)
                acc_testing = utils.get_accuracy(prob_test, batch_y_test)

                # Get accuracy and loss for each network
                acc_check[i] += acc_training
                acc_check_test[i] += acc_testing
                loss_check[i] += loss

            # Get final test prediction
            prob_test_final = output_temp / len(self.networks)

            # Calculate Accuracy (Testing final)
            acc_testing_final = utils.get_accuracy(prob_test_final, batch_y_test)
            acc_check_test_final += acc_testing_final

            if iter % num_print == 0 and iter != 0:
                print(('-------------------------') + ' Epoch: ' + str(iter) + ' -------------------------')
                print('Average Loss(Brier score): ' + str(loss_check / num_print))
                print('Training Accuracy: ' + str(acc_check / num_print))
                print('Testing Accuracy: ' + str(acc_check_test / num_print))
                print('Final Testing Accuracy: ' + str(acc_check_test_final / num_print))
                print('\n')

                loss_check = np.zeros(len(self.networks))
                acc_check = np.zeros(len(self.networks))
                acc_check_test = np.zeros(len(self.networks))
                acc_check_test_final = 0
        
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, default='test', help='Name of the folder the data is saved to inside of results')
    parser.add_argument("-e", "--epoch", type=int, default=5000, help='Number of epochs in training')
    parser.add_argument("-b", "--batch-size", type=int, default=256, help='Batch size in training')
    parser.add_argument("-m", "--model", type=str, default='deep_ensembles_models/sess1', help='Name of the model that will be imported for testing')
    parser.add_argument("-s", "--img-size", type=int, default=28, help='Number of pixels across the img')
    parser.add_argument("-d", "--dataset", type=str, default='mnist', help='The name of the dataset to train')
    parser.add_argument("-l", "--num-label", type=int, default=10, help='Number of classes in the datasets')
    parser.add_argument("-g", "--gpu-fraction", type=float, default=0.5, help='Percentage of the GPU being utilized')
    parser.add_argument("-v", "--validation-ratio", type=float, default=0.1, help='Percentage of the test dataset that will go to validation')
    parser.add_argument("-nn", "--num_net", type=int, default=5, help='Number of networks in the ensemble')
    args = parser.parse_args()
    os.makedirs(f'results/{args.name}', exist_ok=True)
    os.makedirs(f'{args.model}', exist_ok=True)
    Train(name=args.name, epoch=args.epoch, batch_size=args.batch_size, model=args.model, img_size=args.img_size, dataset=args.dataset, num_label=args.num_label, gpu_fraction=args.gpu_fraction, validation_ratio=args.validation_ratio, num_net=args.num_net)
