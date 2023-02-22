#!/bin/bash

#Download gz files
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz


gunzip train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz

curl -LO https://raw.github.com/drj11/pypng/master/code/png.py

mkdir -p datasets

mkdir -p datasets/mnist

mkdir -p datasets/mnist/training

mkdir -p datasets/mnist/testing

cp train-images-idx3-ubyte datasets/mnist/training
cp train-labels-idx1-ubyte datasets/mnist/training
cp t10k-images-idx3-ubyte datasets/mnist/testing
cp t10k-labels-idx1-ubyte datasets/mnist/testing

python convert_mnist_to_png.py datasets/mnist datasets/mnist

rm png.py
rm datasets/mnist/training/*-ubyte* datasets/mnist/testing/*-ubyte*
rm *-ubyte*
