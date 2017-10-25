# mnist_arithmetic

An image dataset that requires performing different arithmetical
operations on digits.

Each image contains a character specifying an operation to be performed, as
well as some number of digits. The corresponding label is whatever one gets
when applying the given operation to the given collection of digits.

The operation to be performed in each image, and the digits to perform them on,
are represented using images from the EMNIST dataset:

Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters. Retrieved from http://arxiv.org/abs/1702.05373.

# Using the dataset:

Download and preprocess EMNIST dataset:
```
python download.py emnist <desired location>
```

Test the dataset:
```
python mnist_arithmetic/dataset.py --path="<desired location>"
```

Then to incorporate into your code, install using

```
python setup.py install
```
and have your code instantiate the class `mnist_arithmetic.MnistArithmeticDataset`.