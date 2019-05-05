# Handwritten-Image-Recognition-Using-Tensorflow

The purpose of this project is created a neural network to recognize hand-written numbers with the MNIST data library, and improve the recognition speed by parallelizing the code with Tensorflow. 

![68747470733a2f2f706970656c696e652e61692f6173736574732f696d672f74656e736f72666c6f772d6c6f676f2d323032783136382e706e67](https://user-images.githubusercontent.com/39222728/57118724-3fa44280-6d33-11e9-8996-a30836242ea4.png)

## Abstract

Image recognition can be a difficult task, but we can work around this with the use of the concept of neural networks. Neural networks allow us to teach an algorithm or "machine" the difference between what is right and wrong, based on paramters we can establish. To scale this application, we can also implement the use of GPU parallel programming like CUDA into our algorithm to speed up applications that might take up alot more processes to complete. 

## Neural Networks 

In its simplest form, a neural network is modeled off the wway our brains work, using synapes to pass electrical signal, or highs and lows. The same can be done to train a model or algorithm what 'right" and "wrong". 

The inputs of a neural network are multiplied by weights, which are in turn added together and given a particular bias. This resulting sum is then passed through an activation function to produce a certain output. Activation function act as a gateway for neurons to send their output from one neuron to the next. The activation function we will be using specifically for this project is the Rectified Linear Unit Function or ReLU function for short. 

Below we can see a simplest flow chart for the process of a simple neural network

![pasted image 0](https://user-images.githubusercontent.com/39222728/57187996-64302400-6ec5-11e9-9add-7753fec4e86a.png)

Neurons within the neural ntowkr are connected in layers. these layers can be catagorized into three sections: inputs neurons, hidden neurons, and outputs neurons. The image below shows us an axample structure of the neruons

![pasted image 0 (1)](https://user-images.githubusercontent.com/39222728/57188035-2253ad80-6ec6-11e9-8c04-de3b5a85635b.png)

Neural network learned through training. During this training a set of pre-labeled data needs to be passed to the network, with its prediction being compared to the actual answer. Eevery time the network is wrong, the network can readjust its weightings for neurons try again until it has converged to its highest accuracy percentage. This training process is the main process in which we will also focus on improving computational time. 

The training set we will use for our neural network will be the MNIST training set, consisting of handwritten digits. We will be using roughly 60,000 training images and 10,000 testing images. 

### Convolutional Neural Networks



## Solving The Problem with Tensorflow 

## Tensorflow's Parallelization Process

## Testing Environment

## Results 
