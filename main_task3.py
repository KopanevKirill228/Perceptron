import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification

from preprocessing import preprocess_data
from perceptron import SingleLayerPerceptron


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def confusion_matrix_values(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    return TP, TN, FP, FN


def precision_score(y_true, y_pred):
    TP, TN, FP, FN = confusion_matrix_values(y_true, y_pred)

    if TP + FP == 0:
        return 0.0

    return TP / (TP + FP)


def recall_score(y_true, y_pred):
    TP, TN, FP, FN = confusion_matrix_values(y_true, y_pred)

    if TP + FN == 0:
        return 0.0

    return TP / (TP + FN)


def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def compute_roc_curve(y_true, y_scores):
    """
    y_true: настоящие метки 0/1
    y_scores: вероятности класса 1
    """
    thresholds = np.sort(np.unique(y_scores))[::-1]

    # Добавляем крайние пороги
    thresholds = np.concatenate(([1.1], thresholds, [-0.1]))

    tpr_values = []
    fpr_values = []

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)

        TP, TN, FP, FN = confusion_matrix_values(y_true, y_pred)

        if TP + FN == 0:
            TPR = 0.0
        else:
            TPR = TP / (TP + FN)

        if FP + TN == 0:
            FPR = 0.0
        else:
            FPR = FP / (FP + TN)

        tpr_values.append(TPR)
        fpr_values.append(FPR)

    return np.array(fpr_values), np.array(tpr_values), thresholds


def roc_auc_score_manual(y_true, y_scores):
    fpr, tpr, thresholds = compute_roc_curve(y_true, y_scores)

    # Сортируем по FPR, чтобы корректно посчитать площадь
    sorted_indices = np.argsort(fpr)
    fpr = fpr[sorted_indices]
    tpr = tpr[sorted_indices]

    auc = np.trapezoid(tpr, fpr)


    return auc


def plot_roc_curve(y_true, y_scores):
    fpr, tpr, thresholds = compute_roc_curve(y_true, y_scores)
    auc = roc_auc_score_manual(y_true, y_scores)

    plt.figure(figsize=(7, 6))

    plt.plot(fpr, tpr, label=f"ROC curve, AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend()
    plt.grid(True)

    plt.show()


def plot_errors(X, y_true, y_pred):
    correct_mask = y_true == y_pred
    error_mask = y_true != y_pred

    plt.figure(figsize=(8, 6))

    plt.scatter(
        X[correct_mask, 0],
        X[correct_mask, 1],
        c=y_true[correct_mask],
        edgecolors="k",
        alpha=0.6,
        label="Correct"
    )

    plt.scatter(
        X[error_mask, 0],
        X[error_mask, 1],
        c="red",
        marker="x",
        s=100,
        label="Errors"
    )

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Misclassified test samples")
    plt.legend()
    plt.grid(True)

    plt.show()


def plot_decision_boundary_with_errors(model, X, y_true, y_pred):
    correct_mask = y_true == y_pred
    error_mask = y_true != y_pred

    plt.figure(figsize=(8, 6))

    plt.scatter(
        X[correct_mask, 0],
        X[correct_mask, 1],
        c=y_true[correct_mask],
        edgecolors="k",
        alpha=0.6,
        label="Correct"
    )

    plt.scatter(
        X[error_mask, 0],
        X[error_mask, 1],
        c="red",
        marker="x",
        s=100,
        label="Errors"
    )

    w1 = model.w[0]
    w2 = model.w[1]
    b = model.b

    x1_min = X[:, 0].min() - 1
    x1_max = X[:, 0].max() + 1

    x1_values = np.linspace(x1_min, x1_max, 100)

    if abs(w2) > 1e-12:
        x2_values = -(w1 * x1_values + b) / w2
        plt.plot(x1_values, x2_values, label="Decision boundary")
    else:
        x_vertical = -b / w1
        plt.axvline(x=x_vertical, label="Decision boundary")

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision boundary and misclassified samples")
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

X_train, X_test, y_train, y_test = preprocess_data(
    X,
    y,
    test_size=0.3,
    random_state=42
)

model = SingleLayerPerceptron(
    n_features=2,
    random_state=42,
    init_type="small_random",
    loss_type="bce",
    l2_lambda=0.0
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

# Вероятности класса 1
y_test_scores = model.forward(X_test)

# Предсказанные классы
y_test_pred = model.predict(X_test)

# Метрики
acc = accuracy(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
auc = roc_auc_score_manual(y_test, y_test_scores)

TP, TN, FP, FN = confusion_matrix_values(y_test, y_test_pred)

print("Test metrics")
print("-" * 40)
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"ROC-AUC:   {auc:.4f}")
print("-" * 40)

print("Confusion matrix values")
print("-" * 40)
print(f"TP: {TP}")
print(f"TN: {TN}")
print(f"FP: {FP}")
print(f"FN: {FN}")
print("-" * 40)

# ROC-кривая
plot_roc_curve(y_test, y_test_scores)

# Ошибочные точки
plot_errors(X_test, y_test, y_test_pred)

# Ошибочные точки + разделяющая граница
plot_decision_boundary_with_errors(model, X_test, y_test, y_test_pred)