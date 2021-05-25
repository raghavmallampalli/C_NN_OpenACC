# C & NN OpenACC

Convolution plus Neural Network written in C/C++ and parallelised in OpenACC

Architecture:
Input > Edge detection convolution > Average Pooling > Sharpen convolution > Average Pooling > 
Custom convolution > Average Pooling > Fully connected hidden layer > Output

256 > 256 > 64 > 64 > 16 > 16 > 4 (flattened to 16) >
10 > 7 (output)

We use simple convolution, pooling and a dense neural network in sequence for the task of classifying microstructures. Our chosen dataset is the NIST Ultra High Carbon Steels micrograph dataset. The microstructure type (pearlite, martensite, etc) is the label to be classified into. 5766 images were extracted from 961 micrographs and the model classifies them into one of the 7 microstructure type labels.

Convolution is performed with Sobel, sharpen and contrast kernels.