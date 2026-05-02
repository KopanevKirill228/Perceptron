import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification

from preprocessing import preprocess_data
from perceptron import SingleLayerPerceptron


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def plot_losses(results, title):
    plt.figure(figsize=(9, 6))

    for label, data in results.items():
        model = data["model"]
        plt.plot(model.val_losses, label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Validation/Test loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.show()


def print_results_table(results):
    print("-" * 80)
    print(
        f"{'Experiment':<25} "
        f"{'Train acc':<12} "
        f"{'Test acc':<12} "
        f"{'||w||':<12} "
        f"{'Final loss':<12}"
    )
    print("-" * 80)

    for label, data in results.items():
        model = data["model"]
        weight_norm = np.linalg.norm(model.w)
        final_loss = model.val_losses[-1]

        print(
            f"{label:<25} "
            f"{data['train_acc']:<12.4f} "
            f"{data['test_acc']:<12.4f} "
            f"{weight_norm:<12.4f} "
            f"{final_loss:<12.4f}"
        )

    print("-" * 80)


def train_model(
    X_train,
    y_train,
    X_test,
    y_test,
    loss_type="bce",
    l2_lambda=0.0,
    lr=0.1,
    epochs=100,
    batch_size=32,
    random_state=42
):
    model = SingleLayerPerceptron(
        n_features=X_train.shape[1],
        random_state=random_state,
        init_type="small_random",
        loss_type=loss_type,
        l2_lambda=l2_lambda
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


def run_bce_vs_hinge_experiment(X_train, X_test, y_train, y_test):
    results = {}

    # BCE использует метки 0/1
    bce_model, bce_train_acc, bce_test_acc = train_model(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        loss_type="bce",
        l2_lambda=0.0,
        lr=0.1,
        epochs=100,
        batch_size=32
    )

    results["BCE"] = {
        "model": bce_model,
        "train_acc": bce_train_acc,
        "test_acc": bce_test_acc
    }

    # Hinge использует метки -1/+1
    y_train_hinge = np.where(y_train == 0, -1, 1)
    y_test_hinge = np.where(y_test == 0, -1, 1)

    hinge_model, hinge_train_acc, hinge_test_acc = train_model(
        X_train=X_train,
        y_train=y_train_hinge,
        X_test=X_test,
        y_test=y_test_hinge,
        loss_type="hinge",
        l2_lambda=0.0,
        lr=0.1,
        epochs=100,
        batch_size=32
    )

    results["Hinge"] = {
        "model": hinge_model,
        "train_acc": hinge_train_acc,
        "test_acc": hinge_test_acc
    }

    print("\nBCE vs Hinge experiment")
    print_results_table(results)
    plot_losses(results, "BCE vs Hinge loss")

    return results


def run_l2_experiment(X_train, X_test, y_train, y_test):
    lambdas = [0.0, 0.001, 0.01, 0.1, 1.0]

    results = {}

    for l2_lambda in lambdas:
        model, train_acc, test_acc = train_model(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            loss_type="bce",
            l2_lambda=l2_lambda,
            lr=0.1,
            epochs=100,
            batch_size=32
        )

        label = f"lambda={l2_lambda}"

        results[label] = {
            "model": model,
            "train_acc": train_acc,
            "test_acc": test_acc
        }

    print("\nL2 regularization experiment")
    print_results_table(results)
    plot_losses(results, "Effect of L2 regularization on BCE loss")

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

bce_hinge_results = run_bce_vs_hinge_experiment(
    X_train,
    X_test,
    y_train,
    y_test
)

l2_results = run_l2_experiment(
    X_train,
    X_test,
    y_train,
    y_test
)