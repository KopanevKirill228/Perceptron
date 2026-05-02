import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification

from preprocessing import preprocess_data
from perceptron import SingleLayerPerceptron


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def train_one_model(
    X_train,
    y_train,
    X_test,
    y_test,
    lr=0.1,
    batch_size=32,
    epochs=100,
    init_type="small_random",
    random_state=42
):
    model = SingleLayerPerceptron(
        n_features=X_train.shape[1],
        random_state=random_state,
        init_type=init_type
    )

    model.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size
    )

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy(y_train, y_train_pred)
    test_acc = accuracy(y_test, y_test_pred)

    return model, train_acc, test_acc


def plot_experiment_losses(results, title):
    plt.figure(figsize=(9, 6))

    for label, data in results.items():
        model = data["model"]
        plt.plot(model.val_losses, label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Validation/Test Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def print_results_table(results):
    print("-" * 55)
    print(f"{'Experiment':<25} {'Train accuracy':<15} {'Test accuracy':<15}")
    print("-" * 55)

    for label, data in results.items():
        print(
            f"{label:<25} "
            f"{data['train_acc']:<15.4f} "
            f"{data['test_acc']:<15.4f}"
        )

    print("-" * 55)


def run_learning_rate_experiment(X_train, y_train, X_test, y_test):
    learning_rates = [0.001, 0.01, 0.5, 1.0]

    results = {}

    for lr in learning_rates:
        model, train_acc, test_acc = train_one_model(
            X_train,
            y_train,
            X_test,
            y_test,
            lr=lr,
            batch_size=32,
            epochs=100,
            init_type="small_random",
            random_state=42
        )

        label = f"lr={lr}"

        results[label] = {
            "model": model,
            "train_acc": train_acc,
            "test_acc": test_acc
        }

    print("\nLearning rate experiment")
    print_results_table(results)
    plot_experiment_losses(results, "Effect of Learning Rate on Loss")

    return results


def run_batch_size_experiment(X_train, y_train, X_test, y_test):
    batch_sizes = [1, 16, 64, 256]

    results = {}

    for batch_size in batch_sizes:
        model, train_acc, test_acc = train_one_model(
            X_train,
            y_train,
            X_test,
            y_test,
            lr=0.1,
            batch_size=batch_size,
            epochs=100,
            init_type="small_random",
            random_state=42
        )

        label = f"batch={batch_size}"

        results[label] = {
            "model": model,
            "train_acc": train_acc,
            "test_acc": test_acc
        }

    print("\nBatch size experiment")
    print_results_table(results)
    plot_experiment_losses(results, "Effect of Batch Size on Loss")

    return results


def run_initialization_experiment(X_train, y_train, X_test, y_test):
    init_types = ["zeros", "small_random", "large_random"]

    results = {}

    for init_type in init_types:
        model, train_acc, test_acc = train_one_model(
            X_train,
            y_train,
            X_test,
            y_test,
            lr=0.1,
            batch_size=32,
            epochs=100,
            init_type=init_type,
            random_state=42
        )

        label = f"init={init_type}"

        results[label] = {
            "model": model,
            "train_acc": train_acc,
            "test_acc": test_acc
        }

    print("\nInitialization experiment")
    print_results_table(results)
    plot_experiment_losses(results, "Effect of Weight Initialization on Loss")

    return results


# =========================
# Основной запуск программы
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

learning_rate_results = run_learning_rate_experiment(
    X_train,
    y_train,
    X_test,
    y_test
)

batch_size_results = run_batch_size_experiment(
    X_train,
    y_train,
    X_test,
    y_test
)

initialization_results = run_initialization_experiment(
    X_train,
    y_train,
    X_test,
    y_test
)