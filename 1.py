import numpy as np
import pandas as pd
from collections import Counter
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# KNN Z METRYKĄ MINKOWSKIEGO
class KNN:
    def __init__(self, k=3, p=2):
        self.k = k
        self.p = p

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            distances = [self._get_distance(x, x_train) for x_train in self.X_train]

            nearest_neighbors = np.argsort(distances)[:self.k]

            nearest_labels = [self.y_train[i] for i in nearest_neighbors]
            most_common_label = Counter(nearest_labels).most_common(1)[0][0]
            y_pred.append(most_common_label)

        return np.array(y_pred)

    def _get_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2) ** self.p) ** (1.0 / self.p)

# METRYKI OCENY
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred):
    classes = np.unique(y_true)
    precisions = []
    for cls in classes:
        TP = np.sum((y_pred == cls) & (y_true == cls))
        FP = np.sum((y_pred == cls) & (y_true != cls))
        if TP + FP == 0:
            precisions.append(0)
        else:
            precisions.append(TP / (TP + FP))
    return np.mean(precisions)

def recall(y_true, y_pred):
    classes = np.unique(y_true)
    recalls = []
    for cls in classes:
        TP = np.sum((y_pred == cls) & (y_true == cls))
        FN = np.sum((y_pred != cls) & (y_true == cls))
        if TP + FN == 0:
            recalls.append(0)
        else:
            recalls.append(TP / (TP + FN))
    return np.mean(recalls)

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    if prec + rec == 0:
        return 0
    return 2 * (prec * rec) / (prec + rec)

# WALIDACJA KRZYŻOWA
def cross_validate_knn(X, y, k_values, p_values, folds=5):
    best_k = None
    best_p = None
    best_acc = 0.0
    all_results = []

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    for p in p_values:
        for k in k_values:
            fold_accuracies = []

            for train_index, val_index in skf.split(X, y):
                X_train_fold, X_val_fold = X[train_index], X[val_index]
                y_train_fold, y_val_fold = y[train_index], y[val_index]

                knn = KNN(k=k, p=p)
                knn.fit(X_train_fold, y_train_fold)
                y_val_pred = knn.predict(X_val_fold)

                fold_accuracies.append(accuracy(y_val_fold, y_val_pred))

            mean_acc = np.mean(fold_accuracies)
            all_results.append((k, p, mean_acc))

            if mean_acc > best_acc:
                best_acc = mean_acc
                best_k = k
                best_p = p

    return best_k, best_p, best_acc, all_results


# MAIN
if __name__ == "__main__":
    data = load_wine()
    X = data.data
    y = data.target

    X_embedded = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=y, palette="viridis")
    plt.title("Wizualizacja t-SNE zbioru Wine")
    plt.legend(title="Klasa")
    plt.show()

    k_values = [1, 3, 5, 7, 9, 11, 13, 15]
    p_values = [1, 2, 3]
    best_k, best_p, best_acc, results = cross_validate_knn(
        X, y,
        k_values=k_values,
        p_values=p_values,
        folds=5
    )

    print("Najlepsze parametry z walidacji krzyżowej:")
    print(" - k =", best_k)
    print(" - p =", best_p)
    print(f"Średnie accuracy (CV) = {best_acc:.4f}")

    df_results = pd.DataFrame(results, columns=["k", "p", "mean_accuracy"])
    pivot = df_results.pivot(index="k", columns="p", values="mean_accuracy")

    plt.figure(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, cmap="viridis")
    plt.title("Accuracy dla różnych k i p (5-fold CV)")
    plt.show()


    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    knn_final = KNN(k=best_k, p=best_p)
    knn_final.fit(X_train, y_train)
    y_pred = knn_final.predict(X_test)

    acc = accuracy(y_test, y_pred)
    prec = precision(y_test, y_pred)
    rec = recall(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nWyniki na zbiorze testowym (k={best_k}, p={best_p}):")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
    disp.plot(cmap="Blues")
    plt.title("Macierz pomyłek dla KNN")
    plt.show()
