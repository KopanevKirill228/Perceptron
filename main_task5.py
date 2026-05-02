import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification

from preprocessing import preprocess_data
from perceptron import SingleLayerPerceptron


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def create_stratified_folds(y, n_splits=5, random_state=42):
    """
    Создаёт stratified k-fold индексы вручную.
    Возвращает список folds, где каждый fold — список индексов.
    """
    random.seed(random_state)

    class_indices = {}

    for i, label in enumerate(y):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(i)

    folds = [[] for _ in range(n_splits)]

    for label in class_indices:
        indices = class_indices[label].copy()
        random.shuffle(indices)

        for i, index in enumerate(indices):
            fold_id = i % n_splits
            folds[fold_id].append(index)

    for fold in folds:
        random.shuffle(fold)

    return folds


def cross_validate(
    X,
    y,
    lr,
    batch_size,
    n_splits=5,
    epochs=100,
    random_state=42
):
    folds = create_stratified_folds(
        y,
        n_splits=n_splits,
        random_state=random_state
    )

    fold_accuracies = []

    all_indices = np.arange(len(y))

    for fold_id in range(n_splits):
        val_indices = np.array(folds[fold_id])

        train_indices = []

        for i in range(n_splits):
            if i != fold_id:
                train_indices.extend(folds[i])

        train_indices = np.array(train_indices)

        X_train_fold = X[train_indices]
        y_train_fold = y[train_indices]

        X_val_fold = X[val_indices]
        y_val_fold = y[val_indices]

        model = SingleLayerPerceptron(
            n_features=X.shape[1],
            random_state=random_state,
            init_type="small_random",
            loss_type="bce",
            l2_lambda=0.0
        )

        model.fit(
            X_train=X_train_fold,
            y_train=y_train_fold,
            X_val=X_val_fold,
            y_val=y_val_fold,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size
        )

        y_val_pred = model.predict(X_val_fold)
        val_accuracy = accuracy(y_val_fold, y_val_pred)

        fold_accuracies.append(val_accuracy)

    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)

    return mean_accuracy, std_accuracy, fold_accuracies


def grid_search_cv(
    X_train,
    y_train,
    learning_rates,
    batch_sizes,
    n_splits=5,
    epochs=100,
    random_state=42
):
    results = []

    for lr in learning_rates:
        for batch_size in batch_sizes:
            mean_acc, std_acc, fold_accs = cross_validate(
                X=X_train,
                y=y_train,
                lr=lr,
                batch_size=batch_size,
                n_splits=n_splits,
                epochs=epochs,
                random_state=random_state
            )

            result = {
                "lr": lr,
                "batch_size": batch_size,
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc,
                "fold_accuracies": fold_accs
            }

            results.append(result)

            print(
                f"lr={lr:<6} "
                f"batch_size={batch_size:<4} "
                f"mean_acc={mean_acc:.4f} "
                f"std={std_acc:.4f}"
            )

    return results


def find_best_params(results):
    """
    Выбираем лучшие параметры по максимальной средней accuracy.
    Если accuracy одинаковая, можно выбрать вариант с меньшим std.
    """
    best_result = results[0]

    for result in results:
        if result["mean_accuracy"] > best_result["mean_accuracy"]:
            best_result = result
        elif result["mean_accuracy"] == best_result["mean_accuracy"]:
            if result["std_accuracy"] < best_result["std_accuracy"]:
                best_result = result

    return best_result


def print_cv_results_table(results):
    print("\nCross-validation results")
    print("-" * 80)
    print(
        f"{'lr':<10} "
        f"{'batch_size':<12} "
        f"{'mean accuracy':<18} "
        f"{'std accuracy':<18} "
        f"{'fold accuracies'}"
    )
    print("-" * 80)

    for result in results:
        fold_values = ", ".join([f"{x:.4f}" for x in result["fold_accuracies"]])

        print(
            f"{result['lr']:<10} "
            f"{result['batch_size']:<12} "
            f"{result['mean_accuracy']:<18.4f} "
            f"{result['std_accuracy']:<18.4f} "
            f"{fold_values}"
        )

    print("-" * 80)


def plot_cv_results(results):
    labels = []
    means = []
    stds = []

    for result in results:
        labels.append(f"lr={result['lr']}\nb={result['batch_size']}")
        means.append(result["mean_accuracy"])
        stds.append(result["std_accuracy"])

    x = np.arange(len(labels))

    plt.figure(figsize=(12, 6))

    plt.bar(x, means, yerr=stds, capsize=5)

    plt.xticks(x, labels, rotation=45)
    plt.ylabel("Mean validation accuracy")
    plt.title("5-fold cross-validation results")
    plt.grid(True, axis="y")

    plt.tight_layout()
    plt.show()


def train_final_model(
    X_train,
    y_train,
    X_test,
    y_test,
    best_lr,
    best_batch_size,
    epochs=100,
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
        lr=best_lr,
        batch_size=best_batch_size
    )

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy(y_train, y_train_pred)
    test_acc = accuracy(y_test, y_test_pred)

    return model, train_acc, test_acc


def plot_final_losses(model):
    plt.figure(figsize=(8, 5))

    plt.plot(model.train_losses, label="Train loss")
    plt.plot(model.val_losses, label="Test loss")

    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.title("Final model loss")
    plt.legend()
    plt.grid(True)

    plt.show()


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

# Сначала делаем обычное train/test разделение.
# Cross-validation будет выполняться только на X_train, y_train.
X_train, X_test, y_train, y_test = preprocess_data(
    X,
    y,
    test_size=0.3,
    random_state=42
)

learning_rates = [0.001, 0.01, 0.1, 0.5, 1.0]
batch_sizes = [1, 16, 32, 64, 256]

cv_results = grid_search_cv(
    X_train=X_train,
    y_train=y_train,
    learning_rates=learning_rates,
    batch_sizes=batch_sizes,
    n_splits=5,
    epochs=100,
    random_state=42
)

print_cv_results_table(cv_results)

plot_cv_results(cv_results)

best_result = find_best_params(cv_results)

best_lr = best_result["lr"]
best_batch_size = best_result["batch_size"]

print("\nBest hyperparameters")
print("-" * 40)
print(f"Best learning rate: {best_lr}")
print(f"Best batch size:    {best_batch_size}")
print(f"Mean accuracy:      {best_result['mean_accuracy']:.4f}")
print(f"Std accuracy:       {best_result['std_accuracy']:.4f}")
print("-" * 40)

final_model, final_train_acc, final_test_acc = train_final_model(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    best_lr=best_lr,
    best_batch_size=best_batch_size,
    epochs=100,
    random_state=42
)

print("\nFinal model results")
print("-" * 40)
print(f"Train accuracy: {final_train_acc:.4f}")
print(f"Test accuracy:  {final_test_acc:.4f}")
print("-" * 40)

plot_final_losses(final_model)