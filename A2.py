import numpy as np
import pandas as pd


def minority_class(labels):
    if len(labels) == 0:
        return 0
    frequencies = labels.value_counts().values  # array, sorted in descending order
    probabilities = [f / len(labels) for f in frequencies[1:]]   # everything except the first class
    impurity = sum(probabilities)
    return impurity


def gini(labels):
    raise NotImplementedError('Your code here')
    return impurity


def entropy(labels):
    raise NotImplementedError('Your code here')
    return impurity


class DTree:
    def __init__(self, metric):
        """Set up a new tree.

        We use the `metric` parameter to supply a impurity function such as Gini or Entropy.
        The other class variables should be set by the "fit" method.
        """
        self._metric = metric  # what are we measuring impurity with? (Gini, Entropy, Minority Class...)
        self._samples = None  # how many training samples reached this node?
        self._distribution = []  # what was the class distribution in this node?
        self._label = None  # What was the majority class of training samples that reached this node?
        self._impurity = None  # what was the impurity at this node?
        self._split = False  # if False, then this is a leaf. If you branch from this node, use this to store the name of the feature you're splitting on.
        self._yes = None  # Holds the "yes" DTree object; None if this is still a leaf node
        self._no = None  # Holds the "no" DTree object; None if this is still a leaf node

    def _best_split(self, features, labels):
        """ Determine the best feature to split on.

        :param features: a pd.DataFrame with named training feature columns
        :param labels: a pd.Series or pd.DataFrame with training labels
        :return: `best_so_far` is a string with the name of the best feature,
        and `best_so_far_impurity` is the impurity on that feature

        For each candidate feature the weighted impurity of the "yes" and "no"
        instances for that feature are computed using self._metric.

        We select the feature with the lowest weighted impurity.
        """
        raise NotImplementedError('Your code here')
        return best_so_far, best_so_far_impurity

    def fit(self, features, labels):
        """ Generate a decision tree by recursively fitting & splitting them

        :param features: a pd.DataFrame with named training feature columns
        :param labels: a pd.Series or pd.DataFrame with training labels
        :return: Nothing.

        First this node is fitted as if it was a leaf node: the training majority label, number of samples,
        class distribution and impurity.

        Then we evaluate which feature might give the best split.

        If there is a best split that gives a lower weighed impurity of the child nodes than the impurity in this node,
        initialize the self._yes and self._no variables as new DTrees with the same metric.
        Then, split the training instance features & labels according to the best splitting feature found,
        and fit the Yes subtree with the instances that split to the True side,
        and the No subtree with the instances that are False according to the splitting feature.
        """
        raise NotImplementedError('Your code here')

        split, split_impurity = self._best_split(features, labels)  # Find the best split, if any

        raise NotImplementedError('... and the rest of your code here')

    def predict(self, features):
        """ Predict the labels of the instances based on the features

        :param features: pd.DataFrame of test features
        :return: predicted labels

        We start by initializing an array of labels where we naively predict this node's label.
        The datatype of this array is set to `object` because otherwise numpy
        might select the minimum needed string length for the current label, regardless of child labels.

        Then if this is not a leaf node, we overwrite those values with the values of Yes and No child nodes,
        based on the feature split in this node.
        """
        results = np.full(features.shape[0], self._label, dtype=object)  # object!!!
        if self._split:  # branch node; recursively replace predictions with child predictions
            yes_index = features[self._split] > 0.5
            results[yes_index] = self._yes.predict(features.loc[yes_index])
            results[~yes_index] = self._no.predict(features.loc[~yes_index])
        return results

    def to_text(self, depth=0):
        if self._split:
            text = f'{"|   " * depth}|---{self._split} = no\n'
            text += self._no.to_text(depth=depth + 1)
            text += f'{"|   " * depth}|---{self._split} = yes\n'
            text += self._yes.to_text(depth=depth + 1)

        else:
            text = f'{"|   " * depth}|---{self._label} ({self._samples})\n'.upper()
        return text


class KFolds:
    def __init__(self, X, y, k, seed=None):
        """ Initialize the KFolds instance

        :param X: pd.DataFrame of feature columns
        :param y: pd.DataFrame or pd.Series of labels
        :param k: number of folds desired
        :param seed: random seed, if you want reproducible results (optional)

        After initialization, self.folds will store k folds.
        Each fold is a pair of arrays with training indices and test indices.
        The folds are as evenly distributed in size as possible.
        All the test segments are pairwise disjoint.
        """
        self.X = X
        self.y = y
        self.k = k
        self.folds = []
        indices = np.arange(X.shape[0])  # integer indices of the instances
        if seed is not None:  # Set random seed if desired.
            np.random.seed(seed=seed)
        np.random.shuffle(indices)  # Shuffle in-place.
        fold_size = X.shape[0] / k  # How many instances per fold? Note that this is a floating point number!
        for fold_num in range(k):
            # The int() is used to handle the floating point numbers and make the segments as equal as possible.
            test = indices[int(fold_num * fold_size):int((fold_num + 1) * fold_size)]
            train = np.concatenate([indices[:int(fold_num * fold_size)], indices[int((fold_num + 1) * fold_size):]])
            self.folds.append((train, test))

    def get_fold(self, fold_num):
        """ Get the training and test data of the k-th fold

        :param fold_num: Which fold's division of the data to use
        :return: Training and test features/labels
        """
        train, test = self.folds[fold_num]  # Select the indices developed for this fold during initialization
        # Use those indices to select instance rows to send to test and training sets.
        X_train = self.X.iloc[train]
        X_test = self.X.iloc[test]
        y_train = self.y.iloc[train]
        y_test = self.y.iloc[test]
        return X_train, X_test, y_train, y_test