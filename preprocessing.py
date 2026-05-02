import random
import numpy as np


def preprocess_data(X, y, test_size=0.3, random_state=42):
    random.seed(random_state)

    # 1. Стратифицированное разделение
    class_indices = {}

    for i, label in enumerate(y):
        if label not in class_indices:
            class_indices[label] = []

        class_indices[label].append(i)

    train_indices = []
    test_indices = []

    for label in class_indices:
        indices = class_indices[label].copy()
        random.shuffle(indices)

        test_count = int(len(indices) * test_size)

        test_indices.extend(indices[:test_count])
        train_indices.extend(indices[test_count:])

    random.shuffle(train_indices)
    random.shuffle(test_indices)

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    # 2. Z-score
    # mean и std считаем только по train
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    std[std == 0] = 1

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test, y_train, y_test