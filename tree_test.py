import pandas as pd


from tree import minority_class, gini, entropy, DTree


def test_minority_class():
    data = pd.Series(['a', 'a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'])
    assert minority_class(data) == 0.4  # simple two-label
    data = pd.Series(['a', 'a', 'a', 'a', 'a', 'c', 'b', 'b', 'b', 'b'])
    assert minority_class(data) == 0.5  # three labels
    data = pd.Series(['a', 'a', 'a', 'c', 'c', 'c', 'b', 'b', 'b', 'b'])
    assert minority_class(data) == 0.6  # no absolute majority
    data = pd.Series([1, 1, -1, -1])
    assert minority_class(data) == 0.5  # even split, numeric labels
    data = pd.Series([1, -1, -1, -1])
    assert minority_class(data) == 0.25  # first instance != majority
    data = pd.Series([1, 'a', -1, -1])
    assert minority_class(data) == 0.5  # mixed data types


def test_gini():
    data = pd.Series(['a', 'a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'])
    assert gini(data) == 0.48  # simple two-label
    data = pd.Series(['a', 'a', 'a', 'a', 'a', 'c', 'b', 'b', 'b', 'b'])
    assert gini(data) == 0.58  # three labels
    data = pd.Series(['a', 'a', 'a', 'c', 'c', 'c', 'b', 'b', 'b', 'b'])
    assert gini(data) == 0.66  # no absolute majority
    data = pd.Series([1, 1, -1, -1])
    assert gini(data) == 0.5  # even split, numeric labels
    data = pd.Series([1, -1, -1, -1])
    assert gini(data) == 0.375  # first instance != majority
    data = pd.Series([1, 'a', -1, -1])
    assert gini(data) == 0.625  # mixed data types


def test_entropy():
    data = pd.Series(['a', 'a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'])
    assert 0.97 < entropy(data) < 0.98  # simple two-label
    data = pd.Series(['a', 'a', 'a', 'a', 'a', 'c', 'b', 'b', 'b', 'b'])
    assert 1.36 < entropy(data) < 1.37  # three labels
    data = pd.Series(['a', 'a', 'a', 'c', 'c', 'c', 'b', 'b', 'b', 'b'])
    assert 1.57 < entropy(data) < 1.58  # no absolute majority
    data = pd.Series([1, 1, -1, -1])
    assert entropy(data) == 1.0  # even split, numeric labels
    data = pd.Series([1, -1, -1, -1])
    assert 0.81 < entropy(data) < 0.82  # first instance != majority
    data = pd.Series([1, 'a', -1, -1])
    assert entropy(data) == 1.5  # mixed data types


# A toy dataset about judging the ripeness of avocados
avocados = pd.DataFrame(data={
        'green':     [1, 1, 1, 1, 0, 0, 0, 0],
        'brown':     [1, 1, 0, 1, 1, 1, 0, 1],
        'firmness':  [1, 1, 0, 1, 1, 0, 1, 1],
        'softness':  [0, 1, 1, 1, 1, 1, 1, 1],
        'nub_loose': [0, 1, 1, 1, 0, 1, 1, 0],
        'ripe':      [0, 1, 0, 1, 0, 0, 1, 1]
})


def test_DTree_best_split_minority():
    # Using minority class
    model = DTree(metric=minority_class)
    X = avocados.iloc[:, :-1]
    y = avocados.iloc[:, -1]
    feature, impurity = model._best_split(X, y)
    assert feature == 'firmness'
    assert impurity == 0.25


def test_DTree_best_split_gini():
    # Using gini
    model = DTree(metric=gini)
    X = avocados.iloc[:, :-1]
    y = avocados.iloc[:, -1]
    feature, impurity = model._best_split(X, y)
    assert feature == 'firmness'
    assert 0.33 < impurity < 0.34


def test_DTree_best_split_entropy():
    # Using entropy
    model = DTree(metric=entropy)
    X = avocados.iloc[:, [0, 1, 3, 4]]  # leave out firmness as feature
    y = avocados.iloc[:, -1]
    feature, impurity = model._best_split(X, y)
    assert feature == 'softness'
    assert 0.86 < impurity < 0.87


def test_DTree_fit_basics():
    """" Check if the fit function filled in these values """
    X = avocados.iloc[:, :-1]
    y = avocados.iloc[:, -1]
    model = DTree(metric=minority_class)
    assert model._label is None, "Before fitting, this should not be set yet."
    assert model._impurity is None, "Before fitting, this should not be set yet."
    assert model._samples is None, "Before fitting, this should not be set yet."
    assert len(model._distribution) == 0, "Before fitting, this should not be set yet."
    model.fit(X, y)
    assert model._label is not None, "After fitting, we should know the majority label in the top node"
    assert model._impurity is not None, "After fitting, we should know the impurity in the top node"
    assert isinstance(model._samples, int), "After fitting, this count how many training samples reached this node"
    assert len(model._distribution) > 0, "After fitting, this should store the frequency of each class in the node"


def test_DTree_fit_children():
    """ Check if the root node has split (it should) and has child nodes """
    X = avocados.iloc[:, :-1]
    y = avocados.iloc[:, -1]
    model = DTree(metric=minority_class)
    assert model._split is False, "Before fitting, this should not be set yet."
    assert model._yes is None, "Before fitting, this should not be set yet."
    assert model._no is None, "Before fitting, this should not be set yet."
    model.fit(X, y)
    assert model._split is not False, "After fitting, the top node should have split"
    assert isinstance(model._yes, DTree), "The Yes child node should be a subtree"
    assert isinstance(model._no, DTree), "The No child node should be a subtree"


def test_DTree_fit_recusively_children():
    """ Check recursively if each node is either a leaf, or split and has two children """
    def recursive(model):
        if model._split:
            assert isinstance(model._yes, DTree), "The Yes child node should be a subtree"
            assert isinstance(model._no, DTree), "The No child node should be a subtree"
            recursive(model._yes)
            recursive(model._no)
    X = avocados.iloc[:, :-1]
    y = avocados.iloc[:, -1]
    model = DTree(metric=minority_class)
    model.fit(X, y)
    recursive(model)


def test_DTree_fit_recusively_child_labels():
    """ Check recursively if each node is labeled """
    def recursive(model):
        assert model._label is not None, "Each node should be labeled."
        if model._split:
            recursive(model._yes)
            recursive(model._no)
    X = avocados.iloc[:, :-1]
    y = avocados.iloc[:, -1]
    model = DTree(metric=minority_class)
    model.fit(X, y)
    recursive(model)


def test_DTree_fit_recusively_decreasing_impurity():
    """ Check if the weighted impurity of children is always lower than that of the parent """
    def recursive_impurity(model):
        if model._split:
            yes_impurity, yes_samples = recursive_impurity(model._yes)
            no_impurity, no_samples = recursive_impurity(model._no)
            weighted_impurity = (yes_impurity * yes_samples) + (no_impurity * no_samples)
            assert weighted_impurity < (model._impurity * model._samples), (
                "The weighted impurity of the children should be smaller than the parent")
        return model._impurity, model._samples
    X = avocados.iloc[:, :-1]
    y = avocados.iloc[:, -1]
    model = DTree(metric=minority_class)
    model.fit(X, y)
    recursive_impurity(model)


def test_DTree_fit_text_string():
    """ Check if we've learned the RIGHT model """
    X = avocados.iloc[:, :-1]
    y = avocados.iloc[:, -1]
    model = DTree(metric=minority_class)
    model.fit(X, y)
    text = model.to_text()
    # Compare the text that you actually got to what it should be:
    assert '\n'+text == ("""
|---firmness = no
|   |---0 (2)
|---firmness = yes
|   |---nub_loose = no
|   |   |---0 (3)
|   |---nub_loose = yes
|   |   |---1 (3)
"""), "The tree should look like this"