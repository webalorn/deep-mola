# DeepMola Neural Network [DML]

This library is built on top of theano.

## Requires:

DML use theano and scipy, and run with python 3

- `python3`
- `numpy`
- `skimage`
- `theano` (do not forget theano dependencies)
- `matplotlib` for plotting network training accuracy

## How to test examples

To launch an example of the `example` folder, you should use:

```python3 main.py <example_name>```

For example, you can launch the mnist example by typing `python3 main.py mnist`. Before starting the example, you download the dataset in the `data` folder.

### Current examples:

- `mnist`

## How to use the library

- Create a model within you python code. After creating the object, you can still modify it, add layers, checkers, etc...
- Build the model (`build` function). After this operation, do not update change the model without re-building
- Train your model with the wanted parameters (`train` function)
- You can then use you model to predict, classify, or whatever you want, using `runSingleEntry` or `runBatch` methods.