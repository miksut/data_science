# Data Science Concepts

This repository hosts some implementations of prominent Data Science concepts. These implementations come in form of Jupyter Notebooks and can be found in the folder `src/`. Concretely, the folder contains the following Notebooks:

- `artificial_neural_network.ipynb`: Composing a neural network that performs the [MNIST](http://yann.lecun.com/exdb/mnist/) classification task. The workflow begins by finding a suitable network and the associated hyperparameters. Then, an exhaustive hyperparameter search is performed to find the optimal configuration. This is followed by fine-tuning the optimally configured network in an attempt to increase its robustness. Finally, the network is evaluated on the test dataset. The implementation relies on the deep learning framework [Keras](https://keras.io/) and the Machine Learning platform [Tensorflow](https://www.tensorflow.org/).

- `generative_vs_discriminative_models.ipynb`: Comparison of discriminative and generative learning as typified by logistic regression and naive Bayes. The comparison is based on the paper by [Andrew Ng and Michael Jordan](https://dl.acm.org/doi/10.5555/2980539.2980648).

- `regression.ipynb`: Implementation of a Linear Regression model using the least squares method. Furthermore, concepts such as regularization, polynomial basis expansion, and cross validation are covered as well.

- `transfer_learning.ipynb`: Extending a convolutional neural network, that has been pretrained on [ImageNet](https://image-net.org/index), with a collection of additional layers (i.e., the custom model) to solve the classification task on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. The workflow includes data augmentation, the convergence of the custom model on the dataset, the fine-tunig of the overall model, and its evaluation on the test dataset. The implementation relies on the deep learning framework [Keras](https://keras.io/) and the Machine Learning platform [Tensorflow](https://www.tensorflow.org/).

The original code is the result of a collaboration with two fellow [UZH](https://www.uzh.ch/en.html) students ([julwil](https://github.com/julwil) and [cdeiac](https://github.com/cdeiac)) and is linked to a lecture offered by the [Data Systems and Theory Group](https://www.ifi.uzh.ch/en/dast.html) in the Department of Informatics at the University of Zurich, Switzerland. This repository contains a slightly revised version of the original code.
