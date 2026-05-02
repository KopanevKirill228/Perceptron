import numpy as np
import matplotlib.pyplot as plt

from preprocessing import preprocess_data
from perceptron import SingleLayerPerceptron

from data_generator import (
    generate_gaussian_data,
    generate_xor_data,
    generate_circle_data
)


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def plot_data(X, y, title):
    plt.figure(figsize=(6, 5))
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_losses(model, title):
    plt.figure(figsize=(7, 5))

    plt.plot(model.train_losses, label="Train loss")
    plt.plot(model.val_losses, label="Test loss")

    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.show()


def plot_decision_boundary(model, X, y, title):
    plt.figure(figsize=(7, 6))

    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k")

    w1 = model.w[0]
    w2 = model.w[1]
    b = model.b

    x1_min = X[:, 0].min() - 0.5
    x1_max = X[:, 0].max() + 0.5

    x1_values = np.linspace(x1_min, x1_max, 100)

    if abs(w2) > 1e-12:
        x2_values = -(w1 * x1_values + b) / w2
        plt.plot(x1_values, x2_values, label="Decision boundary")
    else:
        x_vertical = -b / w1
        plt.axvline(x=x_vertical, label="Decision boundary")

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.show()


def run_experiment(X, y, dataset_name):
    print("\n" + "=" * 60)
    print(dataset_name)
    print("=" * 60)

    plot_data(X, y, f"{dataset_name}: original data")

    X_train, X_test, y_train, y_test = preprocess_data(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    model = SingleLayerPerceptron(
        n_features=2,
        random_state=42,
        init_type="small_random"
    )

    model.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=100,
        lr=0.1,
        batch_size=32
    )

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy(y_train, y_train_pred)
    test_acc = accuracy(y_test, y_test_pred)

    print("Train accuracy:", train_acc)
    print("Test accuracy:", test_acc)

    plot_losses(model, f"{dataset_name}: loss")

    plot_decision_boundary(
        model,
        X_train,
        y_train,
        title=f"{dataset_name}: decision boundary on train data"
    )

    return train_acc, test_acc


# =========================
# 1. Линейно разделимые данные
# =========================

X_gaussian, y_gaussian = generate_gaussian_data(
    n_samples=500,
    center_0=(-2, -2),
    center_1=(2, 2),
    covariance=[[1, 0], [0, 1]],
    noise=0.0,
    random_state=42
)

gaussian_train_acc, gaussian_test_acc = run_experiment(
    X_gaussian,
    y_gaussian,
    "Gaussian linearly separable data"
)


# =========================
# 2. XOR-данные
# =========================

X_xor, y_xor = generate_xor_data(
    n_samples=500,
    noise=0.0,
    random_state=42
)

xor_train_acc, xor_test_acc = run_experiment(
    X_xor,
    y_xor,
    "XOR nonlinear data"
)


# =========================
# 3. Окружность
# =========================

X_circle, y_circle = generate_circle_data(
    n_samples=500,
    radius=0.5,
    noise=0.0,
    random_state=42
)

circle_train_acc, circle_test_acc = run_experiment(
    X_circle,
    y_circle,
    "Circle nonlinear data"
)


# =========================
# Итоговая таблица
# =========================

print("\nFinal results")
print("-" * 70)
print(f"{'Dataset':<35} {'Train accuracy':<15} {'Test accuracy':<15}")
print("-" * 70)

print(
    f"{'Gaussian data':<35} "
    f"{gaussian_train_acc:<15.4f} "
    f"{gaussian_test_acc:<15.4f}"
)

print(
    f"{'XOR data':<35} "
    f"{xor_train_acc:<15.4f} "
    f"{xor_test_acc:<15.4f}"
)

print(
    f"{'Circle data':<35} "
    f"{circle_train_acc:<15.4f} "
    f"{circle_test_acc:<15.4f}"
)

print("-" * 70)