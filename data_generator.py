import numpy as np


def add_label_noise(y, noise=0.0, random_state=42):
    """
    Случайно меняет метку класса с вероятностью noise.
    Для бинарной классификации: 0 -> 1, 1 -> 0.
    """
    np.random.seed(random_state)

    y_noisy = y.copy()

    for i in range(len(y_noisy)):
        if np.random.rand() < noise:
            y_noisy[i] = 1 - y_noisy[i]

    return y_noisy


def generate_gaussian_data(
    n_samples=500,
    center_0=(-2, -2),
    center_1=(2, 2),
    covariance=None,
    noise=0.0,
    random_state=42
):
    """
    Генерирует линейно разделимые данные:
    два гауссовых облака с заданными центрами и ковариацией.
    """
    np.random.seed(random_state)

    if covariance is None:
        covariance = [[1, 0], [0, 1]]

    n_class_0 = n_samples // 2
    n_class_1 = n_samples - n_class_0

    X0 = np.random.multivariate_normal(
        mean=center_0,
        cov=covariance,
        size=n_class_0
    )

    X1 = np.random.multivariate_normal(
        mean=center_1,
        cov=covariance,
        size=n_class_1
    )

    X = np.vstack((X0, X1))
    y = np.array([0] * n_class_0 + [1] * n_class_1)

    y = add_label_noise(y, noise=noise, random_state=random_state)

    return X, y


def generate_xor_data(
    n_samples=500,
    noise=0.0,
    random_state=42
):
    """
    Генерирует XOR-данные.
    Классы расположены по диагоналям:
    класс 0: левый нижний и правый верхний углы
    класс 1: левый верхний и правый нижний углы
    """
    np.random.seed(random_state)

    X = np.random.uniform(-1, 1, size=(n_samples, 2))

    y = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        x1 = X[i, 0]
        x2 = X[i, 1]

        if x1 * x2 < 0:
            y[i] = 1
        else:
            y[i] = 0

    y = add_label_noise(y, noise=noise, random_state=random_state)

    return X, y


def generate_circle_data(
    n_samples=500,
    radius=0.5,
    noise=0.0,
    random_state=42
):
    """
    Генерирует данные типа окружность:
    класс 0 — точки внутри круга,
    класс 1 — точки снаружи круга.
    """
    np.random.seed(random_state)

    X = np.random.uniform(-1, 1, size=(n_samples, 2))

    y = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        x1 = X[i, 0]
        x2 = X[i, 1]

        distance = np.sqrt(x1 ** 2 + x2 ** 2)

        if distance > radius:
            y[i] = 1
        else:
            y[i] = 0

    y = add_label_noise(y, noise=noise, random_state=random_state)

    return X, y