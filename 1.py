import numpy as np
import pandas as pd
from collections import Counter
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# KNN z metryką Minkowskiego
class KNNClassifier:
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

# Metryki oceny
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

# Walidacja krzyżowa
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

                knn = KNNClassifier(k=k, p=p)
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

    k_values = [1, 2, 3, 5, 7, 9, 11, 13, 15]
    p_values = [1, 1.5, 2, 3, 4]
    best_k, best_p, best_acc, results = cross_validate_knn(X, y, k_values=k_values, p_values=p_values, folds=10)
    print("Najlepsze parametry z walidacji krzyżowej:")
    print(" - k =", best_k)
    print(" - p =", best_p)
    print(f"Średnie accuracy (CV) = {best_acc:.4f}")

    df_results = pd.DataFrame(results, columns=["k", "p", "mean_accuracy"])
    pivot_table = df_results.pivot(index="k", columns="p", values="mean_accuracy")
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="viridis")
    plt.xlabel("Wartość p (metryka Minkowskiego)")
    plt.ylabel("Liczba sąsiadów (k)")
    plt.title("Tabela średnich accuracy (CV) dla różnych wartości k i p")
    plt.show()


# Wykres błędnej klasyfikacji w zależności od k dla różnych p 
    df_results = pd.DataFrame(results, columns=["k", "p", "mean_accuracy"])
    pivot = df_results.pivot(index="k", columns="p", values="mean_accuracy")
    plt.figure(figsize=(8, 6))
    for p in pivot.columns:
        plt.plot(pivot.index, 1 - pivot[p], marker='o', label=f'p = {p}')
    plt.xlabel('Liczba sąsiadów (k)')
    plt.ylabel('Błąd (1 - accuracy)')
    plt.title('Błąd klasyfikacji w zależności od k dla różnych metryk Minkowskiego (p)')
    plt.legend(title='Wartość p')
    plt.grid(True)
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    knn_final = KNNClassifier(k=best_k, p=best_p)
    knn_final.fit(X_train, y_train)
    y_pred = knn_final.predict(X_test)
    acc = accuracy(y_test, y_pred)
    prec = precision(y_test, y_pred)
    rec = recall(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Wyniki na zbiorze testowym (k={best_k}, p={best_p}):")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

# Macierz pomyłek
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
    disp.plot(cmap="Blues")
    plt.title("Macierz pomyłek dla KNN")
    plt.show()

# Wizualizacja z wykorzystaniem t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_train_2D = tsne.fit_transform(X_train)

    k_selected = [1, 7]
    p_selected = [1, 2, 4]

    n_rows = len(k_selected)
    n_cols = len(p_selected)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), squeeze=False)

    for i, k_val in enumerate(k_selected):
        for j, p_val in enumerate(p_selected):
            knn_vis = KNNClassifier(k=k_val, p=p_val)
            knn_vis.fit(X_train_2D, y_train)

            x_min, x_max = X_train_2D[:, 0].min() - 1, X_train_2D[:, 0].max() + 1
            y_min, y_max = X_train_2D[:, 1].min() - 1, X_train_2D[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                                 np.linspace(y_min, y_max, 100))
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            Z = knn_vis.predict(grid_points)
            Z = Z.reshape(xx.shape)

            ax = axes[i, j]
            ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
            scatter = ax.scatter(X_train_2D[:, 0], X_train_2D[:, 1], c=y_train, cmap='viridis', edgecolor='k', s=20)
            ax.set_title(f'k = {k_val}, p = {p_val}')
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")

    fig.suptitle("Granice decyzyjne KNN dla różnych wartości k i p w przestrzeni t-SNE", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()