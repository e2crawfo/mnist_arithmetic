# mnist_arithmetic

An image dataset that requires performing different arithmetical
operations on digits.

Each image contains a letter specifying an operation to be performed, as
well as some number of digits. The corresponding label is whatever one gets
when applying the given operation to the given collection of digits.

The operation to be performed in each image, and the digits to perform them on,
are represented using images from the EMNIST dataset.

Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters. Retrieved from http://arxiv.org/abs/1702.05373.

# Instructions

Install:

```
python setup.py install
```

Download and preprocess EMNIST dataset:
```
python download.py emnist <desired location>
```

Test:
```
python mnist_arithmetic/emnist.py <desired location>
```

Incorporate into your code by instantiating the class `mnist_arithmetic.MnistArithmeticDataset`.