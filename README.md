[Uploading README.md…]()
# KNN and Perceptron from scratch

Three simple models implemented with NumPy: a k-nearest-neighbors classifier, a k-nearest-neighbors regressor and a perceptron.

## K-Nearest Neighbors

`KNearestNeighborsClassifier.ipynb` and `KNearestNeighborsRegressor.ipynb`

`fit` only stores the training data. In `predict`, the Euclidean distances between all test points and all training points are computed at once with broadcasting (no Python loop over the points), then the `k` closest training points are taken.

- Classifier: majority vote among the `k` neighbors.
- Regressor: mean of the `k` neighbor values.

Tests:

- Classifier on the two moons dataset (200 points, noise 0.2, `k = 5`), with a plot of the decision regions.
- Regressor on a noisy sine curve (100 points, `k = 5`).

Since the full distance matrix is kept in memory, this is meant for small datasets.

## Perceptron

`Perceptron.ipynb`

- Labels are -1 and +1, the activation is a step function.
- Perceptron learning rule on mini-batches: `w = w + lr * X_b^T (y_b - step(X_b w))`. Only misclassified points change the weights.
- Training stops when all training points are classified correctly, or after `max_iter` epochs.

It is tested on a 2D dataset from `make_classification` (200 points) and the decision boundary is plotted. The two classes in this dataset overlap a little, so the perceptron can not reach 100% and the training accuracy keeps jumping between epochs (roughly between 0.90 and 0.945). This is the expected behavior of the perceptron on data that is not linearly separable.

## Requirements

numpy, matplotlib, scikit-learn (only for generating the test data)
