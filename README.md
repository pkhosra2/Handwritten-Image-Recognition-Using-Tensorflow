
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

Please we the lines of sample code below for importing the MNIST data set

![Capture](https://user-images.githubusercontent.com/39222728/57188355-406fdc80-6ecb-11e9-8ad8-2dcaacdc70b0.JPG)

We can set our MNIST window diimesions as shown bewlow:

![Capture](https://user-images.githubusercontent.com/39222728/57188390-d3a91200-6ecb-11e9-8965-309c70712a83.JPG)

### Convolutional Neural Networks

Convolutional neural networks are useful for image classification because using a regular neural network would be too computationally intensive for this purpose/ For example, if there is a 100x100 image, there would be 10000 pixels which would be treated as neurons.
In order to train the network, we would need connect each of these 10000 neurons to possibly 10000 more neurons in the next layer which would make 100 million required weights to train for just 1 layer

![pastedImage0 (3)](https://user-images.githubusercontent.com/39222728/57188113-27fdc300-6ec7-11e9-998a-ed07896c3603.png)

Therefore, in order to avoid this “Parameter explosion”, it is imperitive that we use a Convolutonal Neural Network. 

A CNN uses a “Sliding Window Function” to reduce the number of parameters to train. This Sliding Window is called a “Kernel” which is usually a 3x3 matrix of weights to multiply to the input. 

The following function will be used to implement this CNN in Python 3:

- new_conv_layer (defines a new convolutional layer)
- new_weights (implements weighting of convolutional layers) 
- new_biases (factors in biases to certain image locations)
- flatten_layer (converts multi-dimensional matrix into flat vector)
- feed_forward ( converts flattened layer into the final output )
- doInference (simulates an evenly spaced group of handwritten characters) 

We can initialize our CNN using hte following lines of code:

![Capture](https://user-images.githubusercontent.com/39222728/57188400-f63b2b00-6ecb-11e9-855e-251af08deb97.JPG)

Below we see a 3D representation of the sldiing window in action

![pastedImage0 (4)](https://user-images.githubusercontent.com/39222728/57188130-7f039800-6ec7-11e9-8420-4d92a362d556.png)

The kernel is slid over the image data a number of times to create many feature maps as output channels. Different shaped kernels(filters) are also applied to learn more and more about the shape of the image. Note that this process is only the first convoltional layer. After the convolutional layer, the image goes through an activation function that we discussed previously. After the activation function, it goes through a max pooling operation which further reduces the parameters required for training. Its important to note that A max pooling operation also uses a kernel and get the highest value from a certain area of data. 

After going through the required amount of convolutional layers(design decision), the output is flattened in order to make the “fully-connected layer”. This layer goes through another transformation called a softmax function which finally outputs probabilities of the image belonging to a classification (eg P(y=”cat”)=0.1)

See the lines of exmaple code below for intializing the fully-connected layer for flattening nad implementing max-pooling

![Capture](https://user-images.githubusercontent.com/39222728/57188411-326e8b80-6ecc-11e9-8e7a-f1ab3ac3b88e.JPG)

![Capture](https://user-images.githubusercontent.com/39222728/57188449-d2c4b000-6ecc-11e9-8696-a6bc829e46b1.JPG)

In the case of digit recognition, we would use 10 softmax functions to find which digit the image probably is. The highest probability out of the 10 wins. The lines of code shown below shows us how to implement the soft-max function

![Capture](https://user-images.githubusercontent.com/39222728/57188455-f982e680-6ecc-11e9-9089-930027807420.JPG)

The final step is training the program to recognize the images correctly. We need to reduce error by iterating through large dataset of training, pre-labeled  images and using a function to correct error. To accomplish this we will use a built-in method called the “Adam Optimizer” which will correct the weights after iterating through labelled images. We can run the the whole process for a number of times called “epochs”. These epochs or iteration can take upwards to hours or even days dependingo on the complexity of the application. 

Below is a snippet of creating the epochs and running through the iterations

![Capture](https://user-images.githubusercontent.com/39222728/57188464-2afbb200-6ecd-11e9-9726-b0d7419b54ef.JPG)

## Solving The Problem with Tensorflow 

Tensorflow is an open source-library for data flow programming, developed by Google. A ‘tensor’ is a data structure similar to that of an array or list. In other words, we can say that it’s the flow of multidimensional matrices. These array will be initialized and manipulated as they are passed along to the Tensorflow graph, the main form of expressing computation. 

The paramters of Tensorflow typically consists of the following parts: 

- Placeholder variables used for inputting data into the graph
- Variables that are going to be optimized so as to make the convolutional network perform better 
- Mathematical formulas for convolutional network 
- A cost measure that can be used to guide the optimization of the variables
- An optimization method which iterates each variable

### Complete Neural Network Flow & Structure 

Below we can see the entire convolutional neural network structure for our approach

![unnamed (1)](https://user-images.githubusercontent.com/39222728/57188291-179b1780-6eca-11e9-8ce3-cbcfdad34d14.jpg)

 We can break down this structure with the following steps:
 
1. Input image is into the algorithm from the MNIST data-set library
2. Convolutional Layer 1: Image is segmented into X segments known as ‘filter weights’ ; the segments are compared using convolution computation with neighbouring pixels 
3. Each computation formulates a ‘channel ‘ 
4. Convolutional Layer 2: Each of X channels is fed through another layer of convolution of X filters 
5. Result is converted (flattened) into a 1-D Vector 
6. Flattened Vector is classified into its respective number (0-9) using the features available 

### Tensor Parameters

Each ‘Tensor’ will have the following dimensions that will make it a 4-dimensional matrix:

- Image number 
- X-axis of each image 
- Y-axis of each image 
- Channels of each image 

Note: input channels may either be colour-channels or filter channels, but since we are feeding grayscale images, we will be using filter channels 

The output is another 4-dimensional tensor with the dimensions: 

- Image number, (same as input)
- X-axis of the image
- Y-axis of the image 
- Channels produced by each of the convolutional filters 

## Training & Testing Our Data

below we see a snippet of codoe used to create placeholder funciton as our tensor as well as first and second convoltional layers

![Capture](https://user-images.githubusercontent.com/39222728/57188438-a6a92f00-6ecc-11e9-966a-db1b308d145b.JPG)


## Testing Environment & System 

Before we parallelize our process, we must take a look at the onboard GPU we are working with:

GPU: Tesla K80 , compute 3.7, having 2496 CUDA cores and 13 SMX (Streaming Multiprocessor Architecture) at 560 MHz speed, 12GB GDDR5 VRAM with 240 GB/s memory bandwidth
PCIe: Host to Device Bandwidth 12 GB/s
CPU: single core hyper threaded Xeon Processors @2.3Ghz i.e(1 core, 2 threads) 

With this GPU we can parallelize 13 x 2496 = 32448 simple operations

## Tensorflow Parallelization

In the python “tf.nn.Conv2d” function, there is a flag “use_cudnn_on_gpu = true” which automatically parallelized the convolution operation

Tensorflow implements the GPU operations in C++ 

This GPU architecture is shown by the diagram below:

![pastedImage0 (5)](https://user-images.githubusercontent.com/39222728/57188524-3ef3e380-6ece-11e9-91cb-367581be7555.png)


## Results 

### Test Sizes & Accuracy

To test our code written we import the Python code into the Google Collaborate environment system

The training and test sample sizes can be seen below:

![pastedImage0 (5)](https://user-images.githubusercontent.com/39222728/57188541-92fec800-6ece-11e9-8d5a-e01c7f004028.png)

From this splot, we are able to achieve the following accuracy, as shown below:

![unnamed (2)](https://user-images.githubusercontent.com/39222728/57188546-b4f84a80-6ece-11e9-88ed-837c9f94cf36.jpg)

After several iterations through epochs, we were able to accurately predict the following set of numbers form the MNIST data set: 

![unnamed (3)](https://user-images.githubusercontent.com/39222728/57188565-deb17180-6ece-11e9-978b-2591ebcb49a9.jpg)

### Impoving Speed With Parallelization


