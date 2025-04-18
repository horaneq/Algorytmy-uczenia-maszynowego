import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ucimlrepo import fetch_ucirepo
import seaborn as sns
from sklearn.manifold import TSNE

banknote_authentication = fetch_ucirepo(id=267)
X = banknote_authentication.data.features
y = banknote_authentication.data.targets

print("Kształt zbioru cech:", X.shape)
print("\nTypy danych:")
print(X.dtypes)

print("\nRozkład klas w zbiorze:")
print(y.value_counts())

print("\nCzy dane są numeryczne?")
print(X.applymap(np.isreal).all())

print("\nStatystyki opisowe:")
print(X.describe())

plt.figure(figsize=(10, 5))
melted = X.melt(var_name="Cecha", value_name="Wartość")
sns.stripplot(data=melted, x="Cecha", y="Wartość", jitter=0.25, alpha=0.4, palette="Set2")
plt.title("Rozrzut wartości cech przed standaryzacją")
plt.ylabel("Wartość")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 4))
sns.countplot(x=y['class'])
plt.title("Rozkład klas w zbiorze")
plt.xlabel("Klasa")
plt.ylabel("Liczba próbek")
plt.tight_layout()
plt.grid(True)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X.values, y.values.ravel(), test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Klasy w y_test:", np.unique(y_test))

tsne1 = TSNE(n_components=2)
X_tsne1 = tsne1.fit_transform(X_train_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_tsne1[:, 0], X_tsne1[:, 1], c=y_train, cmap='coolwarm', edgecolors='k')
plt.title("Zbiór treningowy po t-SNE (2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(*scatter.legend_elements(), title="Klasa")
plt.grid(True)
plt.tight_layout()
plt.show()

class Perceptron:
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-6):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.bias = None
        self.n_iter_ = 0
        self.errors_ = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        y_ = np.where(y == 0, -1, 1)

        for i in range(self.max_iter):
            errors = 0
            prev_weights = self.weights.copy()
            prev_bias = self.bias

            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                predicted = np.where(linear_output >= 0, 1, -1)
                if y_[idx] != predicted:
                    self.weights += self.learning_rate * y_[idx] * x_i
                    self.bias += self.learning_rate * y_[idx]
                    errors += 1

            self.errors_.append(errors)

            if errors == 0:
                print(f"Przerwano w iteracji {i+1} – brak błędów.")
                break

            if np.allclose(self.weights, prev_weights, atol=self.tol) and abs(self.bias - prev_bias) < self.tol:
                print(f"Przerwano w iteracji {i+1} – brak istotnych zmian wag.")
                break

        self.n_iter_ = i + 1

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)


learning_rates = np.logspace(-4, 0, 10)
tolerances = [1e-1, 1e-2, 1e-3, 1e-4, 1e-6]

results = []

for tol in tolerances:
    for lr in learning_rates:
        model = Perceptron(learning_rate=lr, max_iter=1000, tol=tol)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        results.append({
            "tol": tol,
            "learning_rate": lr,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
            "recall": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
            "f1": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
            "iterations": model.n_iter_
        })

df = pd.DataFrame(results)
df["log_lr"] = np.round(np.log10(df["learning_rate"]), 2)
df["log_tol"] = np.round(np.log10(df["tol"]), 2)

for metric in ["accuracy", "precision", "recall", "f1", "iterations"]:
    pivot = df.pivot_table(index="log_tol", columns="log_lr", values=metric)

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, cmap="crest", fmt=".2f", cbar_kws={"label": metric})
    plt.title(f"{metric.capitalize()} w zależności od learning_rate i tol")
    plt.xlabel("log10(Learning Rate)")
    plt.ylabel("log10(Tolerancja)")
    plt.tight_layout()
    plt.show()


tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_train_scaled)

model_tsne = Perceptron(learning_rate=0.01, max_iter=1000)
model_tsne.fit(X_tsne, y_train)

x_min, x_max = X_tsne[:, 0].min() - 1, X_tsne[:, 0].max() + 1
y_min, y_max = X_tsne[:, 1].min() - 1, X_tsne[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = model_tsne.predict(grid).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_train, cmap='coolwarm', edgecolors='k')
plt.title("Granice decyzyjne perceptronu w przestrzeni t-SNE")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.grid(True)
plt.tight_layout()
plt.show()