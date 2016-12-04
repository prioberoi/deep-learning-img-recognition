# Jetson TX1 and Deep Learning Tutorial

This tutorial walks you through setting up your jetson TX1, installing caffe (deep learning framework) prerequisites, installing caffe, and building a deep learning model.

## Jetson TX1

### Setup

The jetson tx1 needs a monitor, mouse and keyboard. Mine came with wifi antennae but I used an ethernet cord while setting up. The most time consuming portion of setting up the Jetson TX1 was flashing over the Jetpack for L4T, which had the Cuda 7+ and cuDNN prerequisites I needed for caffe.

The jetpack needs a 64-bit ubuntu 16.04, which is what is running on the jetson tx1, but I downloaded and install the jetson on a host machine with the same specs first. I used a virtualbox vm on my mac with ubuntu linux x64 (v14.04). The jetpack documentation says it only needs 10gb of space, but my vm needed about 30gb totaly so it wouldn't complain. 

This [guide](http://www.slothparadise.com/setup-cuda-7-0-nvidia-jetson-tx1-jetpack-detailed/) was superbly helpful for installing jetpack. The only thing I would add is that my screen would keep freezing when I was flashing the jetpack over to the jetson. 

Terminal frozen at writing partition app with system.img ![Terminal frozen at writing partition app with system.img](img/scrnshot_frozen_terminal.png)

That turned turned out to be an issue with the USB being set to 1.0 instead of 2.0, so in the USB settings from the VirtualBox manager, I selected "Enable USB Controller" (while the vm was powered down). This required installing the [oracle virtualbox extension pack](https://www.virtualbox.org/wiki/Downloads). Another small snaffu I ran into was my screen would freeze on "applet not running on device, continue with bootloader", which turned out to be an issue because my mac/the vm and my jetson were not on the same network. 

If you're ever stuck at this point, [this](https://developer.ridgerun.com/wiki/index.php?title=Jetpack_output_when_flashing_Tegra_X1) is the output when flashing works.

After flashing over the jetpack and running make, I made sure cuda was installed and tested it out (pretty ocean simulation). I did have to add /usr/local/cuda/bin to my PATH variable in my .bashrc before sourcing it. When I tested my jetson performance I got 318.523 single-precision GFLOP/s at 20 flops per interaction.

## Caffe Installation

After making sure I had the [caffe prerequisites](http://caffe.berkeleyvision.org/installation.html) and cloning caffe into my home directory, I updated Makefile.config by uncommenting `USE_CUDNN := 1` and made sure /usr/include/python2.7 and /usr/lib/python2.7/dist-packages/numpy/core/include exists.

Compiling went pretty smoothly except I got an error on `make all` that it couldn't find hdf5, which worked after I updated `INCLUDE_DIRS` and `LIBRARY_DIRS`
```
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial/
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/aarch64-linux-gnu/hdf5/serial/
```

## Deep Learning

[This tutorial](http://caffe.berkeleyvision.org/gathered/examples/mnist.html) was helpful in training and testing a deep learning model. Since this is my first deep learning model, Michael Nielsen's [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) book was great for wrapping my brain around the concepts that drive deep learning.

The code for the walkthrough of the layers below are from `lenet_train_test.prototxt` in the ~/caffe/examples/mnist directory. 

### Data

There are two data layers, one which will run in the train phase and the other will run in the test phase. 

```
layer {
  name: "mnist"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  transform_param {
    scale: 0.00390625
  }
  data_param {
    source: "examples/mnist/mnist_train_lmdb"
    batch_size: 64
    backend: LMDB
  }
}
layer {
  name: "mnist"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TEST
  }
  transform_param {
    scale: 0.00390625
  }
  data_param {
    source: "examples/mnist/mnist_test_lmdb"
    batch_size: 100
    backend: LMDB
  }
}
```

### MNIST classification model 

Our Convolutional Neural Net is a slight adaption from the LeNet network. It's made up of the following layers.

#### Convolutional Layer

Convolutional neural networks tend to outperform other techniques for image classification, including 2D images like handwritten numbers. This is because they take advantage of the spatial structure of the image. They do this by using local receptive fields, which means a given neuron in a hidden layer is connected to a small region of input neurons and each of those connections has a learned weight and an overall bias. The same weights and overall bias are used for each local receptive field. This means it looks for a particular feature of the image everywhere on the image, which also gives it the flexibility to handle an image where the handwritten digit is in one corner of the image (translation invariance of images). The shared weights and bias for the hidden layer is called a "filter" sometimes.

The map from the input layer to the hidden layer is a feature map. Multiple feature maps make up a convolutional layer. The LeNet-5 convolutional neural net [used 6 feature maps](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf). Our lenet_train_test.prototxt file defines a CNN made of 20 feature maps (`num_output: 20`). 

```
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  param {
    lr_mult: 1 
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 20
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}

```

The `kernel_size: 5` means each local receptive field is 5x5 neurons and a `stride: 1`  means each local receptive field slides one over. 

`lr_mult` is defined first for the filters as a learning rate of 1 and second for the biases, with a learning rate of 2. Our parameters don't include the decay multipliers for each.

The `weight_filler` initializes the filters from [Xavier algorithm](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf) instead of a Gaussian distribution. [This](http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization) and [this](https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/) blogpost was helpful in getting a better understanding of the algorithm. My take away is the Xavier algorithm may help with the issue of a Gaussian distribution weight initialization allowing a neuron to saturate more and therefore see learning slowdown. Caffe does this by picking weights from a Gaussian distribution with a mean of 0 and a variance of 1/N.

#### Pooling Layer

Our pooling layers are used right after the convolutional layers and use a procedure known as max-pooling (`pool: MAX`). The pooling layer takes the output from the hidden neurons in the convolutional layer in regions, in our cases 2 neurons by 2 neurons (`kernel_size: 2`), and returns the maximum activation in that region.  Max-pooling is applied to each feature map. 

```
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}

```

It makes sense to me to think of pooling as a way to ascertain whether a feature was found anywhere in each region. It also reduces the number of parameters down, since each 2x2 neuron in a feature map is reduced to 1 unit (the maximum activation). 

Other pooling methods in caffe are AVE or STOCHASTIC, it may be worth trying each one to see which performs best by looking at the validation results.

#### Fully Connected Layer

This layer produces output as a single with the classification values, which are 0,1,2,3,4,5,6,7,8,9 for the MNIST data. It's fully connected because each neuron in the previous layer is connected to each output neuron
. Caffe calls these Inner Product layers.

```
layer {
  name: "ip1"
  type: "InnerProduct"
  bottom: "pool2"
  top: "ip1"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  inner_product_param {
    num_output: 500
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
```

**This has a `num_output: 500` which is weird because I though this was the number of output classes (i.e. 10, one for each possible digit)**

#### ReLU activations for the neurons

Interestingly, rectified linear activation tend to outperform sigmoid or tanh functions, but I couldn't find a solid reason why. When we look at optimizing our model, we might try to compare the three.

The ReLU activation function takes max(0,z). We're going to train for 60 epochs, **with a learning rate (`lr_mult` pr `blobs_lr`) of 0.03 and an l2 regularization parameter (`weight_decay`) of 0.1.**

```
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "ip1"
  top: "ip1"
}
```

### The accuracy layer

This is only run in the test phase, it will report out the model accuracy every 100 iterations.

```
layer {
  name: "accuracy"
  type: "Accuracy"
  bottom: "ip2"
  bottom: "label"
  top: "accuracy"
  include {
    phase: TEST
  }
}
```

#### The loss layer

```
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "ip2"
  bottom: "label"
  top: "loss"
}
```

#### The whole network

The `input --> convolutional layer --> pooling layer --> fully connected layer --> output layer` is a common set up in convolutional networks. 

The convolutional layer and pooling layers look for local features and the fully connected layer integrates what has been learned across the whole image.

Our network goes `input --> convolutional layer --> pooling layer --> convolutional layer --> pooling layer --> fully connected layer --> fully connected layer`

By adding that second round of convolutional and pooling layers, we are taking the output from the first pooling layer (each neuron of which indicates the presence or absence of a particular feature), which makes sense if you think of the output from that layer as a condensed down version of the original image. Also, this second convolutional layer doesn't just work on the 5 x 5 local receptive field from one feature map, but from all feature maps.

### Training the model

The `lenet_solver.prototxt` defines is where we define variables for ths olver. 

where to find the lenet_train_test.prototxt file
```
# The train/test net protocol buffer definition
net: "examples/mnist/lenet_train_test.prototxt"
```


```
# test_iter specifies how many forward passes the test should carry out.
# In the case of MNIST, we have test batch size 100 and 100 test iterations,
# covering the full 10,000 testing images.
test_iter: 100

```
The code above prints the training loss function every 100 iterations, and below, tests the network every 500 iterations

```
# Carry out testing every 500 training iterations.
test_interval: 500
```

This defines a learning rate of 0.01 and a momentum of 0.9. For SGD, caffe recommends that the learning rate be around 0.01 and that it drops by a constant factor (`gamma`) throughout training
```
# The base learning rate, momentum and the weight decay of the network.
base_lr: 0.01
momentum: 0.9
weight_decay: 0.0005
# The learning rate policy
lr_policy: "inv"
gamma: 0.0001
power: 0.75
```


```
# Display every 100 iterations
display: 100
# The maximum number of iterations
max_iter: 10000
# snapshot intermediate results
snapshot: 5000
snapshot_prefix: "examples/mnist/lenet"
# solver mode: CPU or GPU
solver_mode: GPU
```

Next we run the train_lenet.sh file to train the model. This runs the train command and passes the protobuf file as an argument.

```
./examples/mnist/train_lenet.sh
```

That took about 3 minutes.

### Results

We got an accuracy of 0.9905, and a testing loss function of 0.0299812.

That's pretty great. 

The actual model was stored to examples/mnist/lenet_iter_10000.caffemodel as a binary proto file.

The solver state was saved to examples/mnist/lenet_iter_10000.solverstate.

We also got the model and solver state at 5000.

### Optimizing

Things to try

- Sigmoid, tanh and Rectified Linear Unit (ReLU) activation functions are options for activation functions.

convolutional layer
- change number of features to 40 feature maps: num_output: 20
- change kernel_size in convolutional layer
- play with strid length
- change weight initialization (http://neuralnetworksanddeeplearning.com/chap3.html#weight_initialization) from xavier to normalized gaussian random variables (which means a neuron can saturate and have a learning slowdown):
	http://caffe.berkeleyvision.org/tutorial/layers.html
	weight_filler {
      type: "gaussian" # initialize the filters from a Gaussian
      std: 0.01        # distribution with stdev 0.01 (default mean: 0)
    }

regularization:
- L1 and L2 regularization
	try l2 regularization of λ=0.1
- dropout 
- artificial expansion of the training data

loss:
- backpropagation

- try a learning rate of η=0.03
- could the learning rate be a function of the epoch number?

Here is a plot of the per-epoch validaion accuracies for each.
