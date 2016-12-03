
###########
### commands
###########

cd $CAFFE_ROOT
# download the data
./data/mnist/get_mnist.sh
# coverts the mnist data into lmdb format (depends on the value assigned to $BACKEND)
./examples/mnist/create_mnist.sh
# this created two folders:
# ~/caffe/examples/mnist/mnist_test_lmdb
# ~/caffe/examples/mnist/mnist_train_lmdb

# run the train_lenet.sh file to train the model
./examples/mnist/train_lenet.sh
