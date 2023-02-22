#!/bin/bash

#Download git repo
git clone https://github.com/davidflanagan/notMNIST-to-MNIST.git

#Making directories for final images
mkdir -p datasets
mkdir -p datasets/not_mnist
mkdir -p datasets/not_mnist/training
mkdir -p datasets/not_mnist/testing

cd notMNIST-to-MNIST/

pwd

gunzip train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz

ls

cp train-images-idx3-ubyte ../datasets/not_mnist/training
cp train-labels-idx1-ubyte ../datasets/not_mnist/training
cp t10k-images-idx3-ubyte ../datasets/not_mnist/testing
cp t10k-labels-idx1-ubyte ../datasets/not_mnist/testing


cd ../

curl -LO https://raw.github.com/drj11/pypng/master/code/png.py

python convert_mnist_to_png.py datasets/not_mnist datasets/not_mnist
pwd

mv datasets/not_mnist/training/0/ datasets/not_mnist/training/a/
mv datasets/not_mnist/training/1/ datasets/not_mnist/training/b/
mv datasets/not_mnist/training/2/ datasets/not_mnist/training/c/
mv datasets/not_mnist/training/3/ datasets/not_mnist/training/d/
mv datasets/not_mnist/training/4/ datasets/not_mnist/training/e/
mv datasets/not_mnist/training/5/ datasets/not_mnist/training/f/
mv datasets/not_mnist/training/6/ datasets/not_mnist/training/g/
mv datasets/not_mnist/training/7/ datasets/not_mnist/training/h/
mv datasets/not_mnist/training/8/ datasets/not_mnist/training/i/
mv datasets/not_mnist/training/9/ datasets/not_mnist/training/j/

mv datasets/not_mnist/testing/0/ datasets/not_mnist/testing/a/
mv datasets/not_mnist/testing/1/ datasets/not_mnist/testing/b/
mv datasets/not_mnist/testing/2/ datasets/not_mnist/testing/c/
mv datasets/not_mnist/testing/3/ datasets/not_mnist/testing/d/
mv datasets/not_mnist/testing/4/ datasets/not_mnist/testing/e/
mv datasets/not_mnist/testing/5/ datasets/not_mnist/testing/f/
mv datasets/not_mnist/testing/6/ datasets/not_mnist/testing/g/
mv datasets/not_mnist/testing/7/ datasets/not_mnist/testing/h/
mv datasets/not_mnist/testing/8/ datasets/not_mnist/testing/i/
mv datasets/not_mnist/testing/9/ datasets/not_mnist/testing/j/


#Cleanup
rm -rf notMNIST-to-MNIST/
rm png.py
