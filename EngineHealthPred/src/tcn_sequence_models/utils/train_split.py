def train_test_split(sequences_X, sequences_y):
    """Splits the sequences into training data.

    :param sequences_X: input data sequences
    :param sequences_y: target data sequences
    :return: the training data
    """
    X_train = []
    y_train = []
    for el in sequences_X:
        X_train.append(el)

    y_train = sequences_y
    # for el_X, el_y in zip(sequences_X, sequences_y):
    #     X_train.append(el_X)
    #     y_train.append(el_y)

    return X_train, y_train