import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification

from preprocessing import preprocess_data
from perceptron import SingleLayerPerceptron


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def train_model(
    X_train,
    y_train,
    X_test,
    y_test,
    use_momentum=False,
    beta=0.9,
    lr=0.1,
    epochs=100,
    batch_size=32,
    random_state=42
):
    model = SingleLayerPerceptron(
        n_features=X_train.shape[1],
        random_state=random_state,
        init_type="small_random",
        loss_type="bce",
        l2_lambda=0.0
    )

    model.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        use_momentum=use_momentum,
        beta=beta
    )

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy(y_train, y_train_pred)
    test_acc = accuracy(y_test, y_test_pred)

    return model, train_acc, test_acc


def print_results_table(results):
    print("-" * 75)
    print(
        f"{'Optimizer':<20} "
        f"{'Train acc':<12} "
        f"{'Test acc':<12} "
        f"{'Final train loss':<18} "
        f"{'Final test loss':<18}"
    )
    print("-" * 75)

    for label, data in results.items():
        model = data["model"]

        print(
            f"{label:<20} "
            f"{data['train_acc']:<12.4f} "
            f"{data['test_acc']:<12.4f} "
            f"{model.train_losses[-1]:<18.4f} "
            f"{model.val_losses[-1]:<18.4f}"
        )

    print("-" * 75)


def plot_train_losses(results):
    plt.figure(figsize=(9, 6))

    for label, data in results.items():
        model = data["model"]
        plt.plot(model.train_losses, label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.title("Train loss: SGD vs Momentum")
    plt.legend()
    plt.grid(True)

    plt.show()


def plot_test_losses(results):
    plt.figure(figsize=(9, 6))

    for label, data in results.items():
        model = data["model"]
        plt.plot(model.val_losses, label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Test loss")
    plt.title("Test loss: SGD vs Momentum")
    plt.legend()
    plt.grid(True)

    plt.show()


def run_momentum_experiment(X_train, y_train, X_test, y_test):
    results = {}

    # Обычный SGD без momentum
    model, train_acc, test_acc = train_model(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        use_momentum=False,
        beta=0.0,
        lr=0.1,
        epochs=100,
        batch_size=32,
        random_state=42
    )

    results["SGD"] = {
        "model": model,
        "train_acc": train_acc,
        "test_acc": test_acc
    }

    # Momentum с разными beta
    betas = [0.5, 0.9, 0.99]

    for beta in betas:
        model, train_acc, test_acc = train_model(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            use_momentum=True,
            beta=beta,
            lr=0.1,
            epochs=100,
            batch_size=32,
            random_state=42
        )

        label = f"Momentum beta={beta}"

        results[label] = {
            "model": model,
            "train_acc": train_acc,
            "test_acc": test_acc
        }

    print("\nMomentum experiment")
    print_results_table(results)

    plot_train_losses(results)
    plot_test_losses(results)

    return results


# =========================
# Основной запуск
# =========================

X, y = make_classification(
    n_samples=500,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42
)

X_train, X_test, y_train, y_test = preprocess_data(
    X,
    y,
    test_size=0.3,
    random_state=42
)

momentum_results = run_momentum_experiment(
    X_train,
    y_train,
    X_test,
    y_test
)