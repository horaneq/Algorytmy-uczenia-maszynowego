import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix


class Perceptron:
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-6):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        y_ = np.where(y == 0, -1, 1)

        for _ in range(self.max_iter):
            errors = 0
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                predicted = np.where(linear_output >= 0, 1, -1)
                if y_[idx] != predicted:
                    self.weights += self.learning_rate * y_[idx] * x_i
                    self.bias += self.learning_rate * y_[idx]
                    errors += 1
            if errors == 0:
                break

    def predict_proba(self, X):
        return np.dot(X, self.weights) + self.bias

    def predict(self, X):
        return self.predict_proba(X)


class OVRClassifier:
    def __init__(self, base_classifier):
        self.base_classifier = base_classifier
        self.classifiers = {}
        self.classes_ = []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for cls in self.classes_:
            y_binary = np.where(y == cls, 1, 0)
            clf = self.base_classifier()
            clf.fit(X, y_binary)
            self.classifiers[cls] = clf

    def predict(self, X):
        margins = []
        for cls in self.classes_:
            clf = self.classifiers[cls]
            if hasattr(clf, "decision_function"):
                margin = clf.decision_function(X)
            else:
                margin = clf.predict_proba(X)
            margins.append(margin)
        margins_matrix = np.column_stack(margins)
        return self.classes_[np.argmax(margins_matrix, axis=1)]


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision_macro(y_true, y_pred):
    classes = np.unique(y_true)
    precisions = []
    for cls in classes:
        TP = np.sum((y_pred == cls) & (y_true == cls))
        FP = np.sum((y_pred == cls) & (y_true != cls))
        precisions.append(TP / (TP + FP) if TP + FP > 0 else 0)
    return np.mean(precisions)

def recall_macro(y_true, y_pred):
    classes = np.unique(y_true)
    recalls = []
    for cls in classes:
        TP = np.sum((y_pred == cls) & (y_true == cls))
        FN = np.sum((y_pred != cls) & (y_true == cls))
        recalls.append(TP / (TP + FN) if TP + FN > 0 else 0)
    return np.mean(recalls)

def f1_macro(y_true, y_pred):
    prec = precision_macro(y_true, y_pred)
    rec = recall_macro(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec) if prec + rec > 0 else 0

def precision_micro(y_true, y_pred):
    TP = np.sum(y_true == y_pred)
    return TP / len(y_pred)

def recall_micro(y_true, y_pred):
    TP = np.sum(y_true == y_pred)
    return TP / len(y_true)

def f1_micro(y_true, y_pred):
    prec = precision_micro(y_true, y_pred)
    rec = recall_micro(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec) if prec + rec > 0 else 0

data = load_wine()
X = data.data
y = data.target

df = pd.DataFrame(X, columns=data.feature_names)
df['class'] = y

plt.figure(figsize=(6, 4))
sns.countplot(x='class', data=df, palette='Set2')
plt.title("Rozkład klas w zbiorze")
plt.xlabel("Klasa")
plt.ylabel("Liczba próbek")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
melted = df.melt(id_vars='class', var_name='Cecha', value_name='Wartość')
sns.stripplot(data=melted, x='Cecha', y='Wartość', jitter=0.25, alpha=0.4, palette='Set2')
plt.title("Rozrzut wartości cech przed standaryzacją")
plt.ylabel("Wartość")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_perceptron = []
scores_svc = []
cms_perceptron = []
cms_svc = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y), 1):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    ovr_p = OVRClassifier(lambda: Perceptron())
    ovr_p.fit(X_train, y_train)
    y_pred_p = ovr_p.predict(X_test)

    ovr_svc = OVRClassifier(lambda: SVC(kernel="linear"))
    ovr_svc.fit(X_train, y_train)
    y_pred_svc = ovr_svc.predict(X_test)

    scores_perceptron.append([
        accuracy(y_test, y_pred_p),
        precision_macro(y_test, y_pred_p),
        recall_macro(y_test, y_pred_p),
        f1_macro(y_test, y_pred_p),
        precision_micro(y_test, y_pred_p),
        recall_micro(y_test, y_pred_p),
        f1_micro(y_test, y_pred_p),
    ])

    scores_svc.append([
        accuracy(y_test, y_pred_svc),
        precision_macro(y_test, y_pred_svc),
        recall_macro(y_test, y_pred_svc),
        f1_macro(y_test, y_pred_svc),
        precision_micro(y_test, y_pred_svc),
        recall_micro(y_test, y_pred_svc),
        f1_micro(y_test, y_pred_svc),
    ])

    cm_p = confusion_matrix(y_test, y_pred_p)
    cm_s = confusion_matrix(y_test, y_pred_svc)
    cms_perceptron.append(cm_p)
    cms_svc.append(cm_s)

    print(f"\n[Fold {fold}] Perceptron:")
    print(f"Accuracy={scores_perceptron[-1][0]:.3f}, "
          f"Prec_macro={scores_perceptron[-1][1]:.3f}, Rec_macro={scores_perceptron[-1][2]:.3f}, "
          f"F1_macro={scores_perceptron[-1][3]:.3f}, Prec_micro={scores_perceptron[-1][4]:.3f}, "
          f"Rec_micro={scores_perceptron[-1][5]:.3f}, F1_micro={scores_perceptron[-1][6]:.3f}")

    print(f"[Fold {fold}] SVC:")
    print(f"Accuracy={scores_svc[-1][0]:.3f}, "
          f"Prec_macro={scores_svc[-1][1]:.3f}, Rec_macro={scores_svc[-1][2]:.3f}, "
          f"F1_macro={scores_svc[-1][3]:.3f}, Prec_micro={scores_svc[-1][4]:.3f}, "
          f"Rec_micro={scores_svc[-1][5]:.3f}, F1_micro={scores_svc[-1][6]:.3f}")

sum_cm_perceptron = np.sum(cms_perceptron, axis=0)
sum_cm_svc = np.sum(cms_svc, axis=0)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(sum_cm_perceptron, annot=True, fmt='d', ax=axes[0], cmap="Blues",
            xticklabels=data.target_names, yticklabels=data.target_names)
axes[0].set_title("Suma macierzy błędów Perceptron (OVR) - CV")
axes[0].set_xlabel("Predykcje")
axes[0].set_ylabel("Prawdziwe klasy")

sns.heatmap(sum_cm_svc, annot=True, fmt='d', ax=axes[1], cmap="Greens",
            xticklabels=data.target_names, yticklabels=data.target_names)
axes[1].set_title("Suma macierzy błędów SVC (OVR) - CV")
axes[1].set_xlabel("Predykcje")
axes[1].set_ylabel("Prawdziwe klasy")

plt.tight_layout()
plt.show()

metrics = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)", 
           "Precision (micro)", "Recall (micro)", "F1 (micro)"]

mean_perceptron = np.mean(scores_perceptron, axis=0)
mean_svc = np.mean(scores_svc, axis=0)

df_comparison = pd.DataFrame([mean_perceptron, mean_svc], 
                             index=["Perceptron", "SVC"], 
                             columns=metrics)

plt.figure(figsize=(10, 4))
sns.heatmap(df_comparison, annot=True, cmap="YlGnBu", fmt=".3f")
plt.title("Porównanie metryk: Perceptron vs SVC (OVR, 5-fold CV)")
plt.tight_layout()
plt.show()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

ovr_perceptron = OVRClassifier(lambda: Perceptron(max_iter=5000))
ovr_perceptron.fit(X_scaled, y)

ovr_svc = OVRClassifier(lambda: SVC(kernel="linear"))
ovr_svc.fit(X_scaled, y)

x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
grid_2d = np.c_[xx.ravel(), yy.ravel()]

grid_13d = pca.inverse_transform(grid_2d)

Z_perceptron = ovr_perceptron.predict(grid_13d).reshape(xx.shape)
Z_svc = ovr_svc.predict(grid_13d).reshape(xx.shape)

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z_perceptron, alpha=0.3, cmap=cmap_light)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette=cmap_bold.colors,
                edgecolor='k', s=50)
plt.title("Granice decyzyjne Perceptron (OVR) na PCA")
plt.xlabel("PCA wymiar 1")
plt.ylabel("PCA wymiar 2")
plt.legend(title="Klasa")

plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z_svc, alpha=0.3, cmap=cmap_light)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette=cmap_bold.colors,
                edgecolor='k', s=50)
plt.title("Granice decyzyjne SVC (OVR) na PCA")
plt.xlabel("PCA wymiar 1")
plt.ylabel("PCA wymiar 2")
plt.legend(title="Klasa")

plt.tight_layout()
plt.show()

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

ovr_p_tsne = OVRClassifier(lambda: Perceptron(max_iter=1000))
ovr_p_tsne.fit(X_tsne, y)

ovr_s_tsne = OVRClassifier(lambda: SVC(kernel="linear"))
ovr_s_tsne.fit(X_tsne, y)

# Siatka punktów
x_min, x_max = X_tsne[:, 0].min() - 1, X_tsne[:, 0].max() + 1
y_min, y_max = X_tsne[:, 1].min() - 1, X_tsne[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]

# Predykcje
Z_perceptron_tsne = ovr_p_tsne.predict(grid).reshape(xx.shape)
Z_svc_tsne = ovr_s_tsne.predict(grid).reshape(xx.shape)

# Wizualizacja
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z_perceptron_tsne, alpha=0.3, cmap=cmap_light)
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette=cmap_bold.colors,
                edgecolor='k', s=50)
plt.title("Granice decyzyjne Perceptron (OVR) w przestrzeni t-SNE")
plt.xlabel("t-SNE wymiar 1")
plt.ylabel("t-SNE wymiar 2")
plt.legend(title="Klasa")

plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z_svc_tsne, alpha=0.3, cmap=cmap_light)
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette=cmap_bold.colors,
                edgecolor='k', s=50)
plt.title("Granice decyzyjne SVC (OVR) w przestrzeni t-SNE")
plt.xlabel("t-SNE wymiar 1")
plt.ylabel("t-SNE wymiar 2")
plt.legend(title="Klasa")

plt.tight_layout()
plt.show()