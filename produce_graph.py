import csv
import matplotlib.pyplot as plt
import pandas as pd

def main():
    mnist_fashion_train = produce_data('mnist/2070_2/train_fashion_results.csv')
    mnist_fashion = produce_data('mnist/2070_2/non_train_fashion_results.csv')
    #mnist_cifar10 = produce_data('mnist/2070/non_train_cifar10_results.csv')
    fashion_mnist_train = produce_data('fashion/2070_2/train_mnist_results.csv')
    fashion_mnist = produce_data('fashion/2070_2/non_train_mnist_results.csv')
    #fashion_cifar10 = produce_data('fashion/2070/non_train_cifar10_results.csv')
    
    
    print('Train: mnist')
    print(mnist_fashion_train)
    print('Train: mnist\tNon: fashion')
    print(mnist_fashion)
    #print('Train: mnist\tNon: cifar10')
    #print(mnist_cifar10)
    print('Train: fashion')
    print(fashion_mnist_train)
    print('Train: fashion\tNon: mnist')
    print(fashion_mnist)
    #print('Train: fashion\tNon: cifar10')
    #print(fashion_cifar10)
    

def produce_data(path):
    mean = []
    std = []
    fp_single = []
    fp_ensemble = []
    value = 0.9
    for i in range(1,11):
        df = pd.read_csv(f'results/simple/{i}/'+path)
        mean.append(df['time(s)'].mean())
        std.append(df['time(s)'].std())
        fp_single.append(df['single_prob'].gt(value).sum())
        fp_ensemble.append(df['ensemble_prob'].gt(value).sum())
    
    data = pd.DataFrame(data={'time_avg': mean, 'time_std': std, 'fp_single': fp_single, 'fp_ensemble': fp_ensemble})
    return data

if __name__=='__main__':
    main()
