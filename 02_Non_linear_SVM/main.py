import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.svm import SVC
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.preprocessing import MinMaxScaler

X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=0.5,
                           random_state=8)
# Ja no necessitem canviar les etiquetes, Scikit ho fa per nosaltres

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Els dos algorismes es beneficien d'estandaritzar les dades

scaler = MinMaxScaler()  # StandardScaler()
X_transformed = scaler.fit_transform(X_train)
X_test_transformed = scaler.transform(X_test)

gamma = 1.0 / (X_transformed.shape[1] * X_transformed.var())


# Feina 1

def kernel_lineal(x1, x2):
    return x1.dot(x2.T)


svm = SVC(C=1000.0, kernel="linear")
svm.fit(X_transformed, y_train)

svm2 = SVC(C=1000.0, kernel=kernel_lineal)
svm2.fit(X_transformed, y_train)

print("Precisión de SVM con kernel lineal original: ", precision_score(y_test, svm.predict(X_test_transformed)))
print("Precisión de SVM con kernel lineal custom: ", precision_score(y_test, svm2.predict(X_test_transformed)))


# Feina 2

def kernel_gauss(x1, x2):
    return np.exp(-gamma * distance_matrix(x1, x2) ** 2)


svm = SVC(C=1000.0, kernel="rbf")
svm.fit(X_transformed, y_train)

svm2 = SVC(C=1000.0, kernel=kernel_gauss)
svm2.fit(X_transformed, y_train)

print("Precisión de SVM con kernel gaussiano original: ", precision_score(y_test, svm.predict(X_test_transformed)))
print("Precisión de SVM con kernel gaussiano custom: ", precision_score(y_test, svm2.predict(X_test_transformed)))


# Feina 3
def kernel_poly(x1, x2, degree=3):
    return (gamma * kernel_lineal(x1, x2) + 0) ** degree


svm = SVC(C=1000.0, kernel="poly")
svm.fit(X_transformed, y_train)

svm2 = SVC(C=1000.0, kernel=kernel_poly)
svm2.fit(X_transformed, y_train)

print("Precisión de SVM con kernel polinómico original: ", precision_score(y_test, svm.predict(X_test_transformed)))
print("Precisión de SVM con kernel polinómico custom: ", precision_score(y_test, svm2.predict(X_test_transformed)))


# Bucle
kernels = {"linear": kernel_lineal, "rbf": kernel_gauss, "poly": kernel_poly}
for kernel in kernels:
    def kernel_poly(x1, x2, degree=3):
        return (gamma * kernel_lineal(x1, x2) + 0) ** degree


    svm = SVC(C=1000.0, kernel=kernel)
    svm.fit(X_transformed, y_train)

    svm2 = SVC(C=1000.0, kernel=kernels[kernel])
    svm2.fit(X_transformed, y_train)

    print("Precisión de SVM con kernel polinómico original: ", precision_score(y_test, svm.predict(X_test_transformed)))
    print("Precisión de SVM con kernel polinómico custom: ", precision_score(y_test, svm2.predict(X_test_transformed)))
    disp = DecisionBoundaryDisplay.from_estimator(svm2, X_transformed,
                                                  response_method="predict",
                                                  xlabel="c1",
                                                  ylabel="c2",
                                                  alpha=0.5)

    disp.ax_.scatter(X_transformed[:, 0], X_transformed[:, 1], edgecolors='k')

    plt.show()
