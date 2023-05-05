import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import accuracy_score


def confusion_matrix(y_true, y_pred):

    labels = sorted(np.unique(y_true))
    n = len(labels)

    cm = np.zeros((n, n), dtype='int')
    
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            cm[i, j] = np.sum(np.logical_and(y_pred == pred_label, y_true == true_label))

    return cm


def confusion_matrix_plot(y_test, y_pred, ax=None):

    cm = confusion_matrix(y_test, y_pred)
    labels = np.unique(y_test)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', linewidths=0.5, square=True, cmap = 'mako',
                xticklabels=labels, yticklabels=labels, ax=ax, cbar=True)

    ax.set_title(f'Accuracy Score: {np.sum(np.diag(cm)) / np.sum(cm):.2f}', size=15)
    ax.set_ylabel('True label', size=12)
    ax.set_xlabel('Predicted label', size=12)

    return ax


class all:

    def __init__(self, digits):
        self.digits = sorted(digits)


    def acurracy_all(self, y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)


    def error_all(self, y_true, y_pred):
        return 1 - accuracy_score(y_true, y_pred)


    def precision_all(self, y_true, y_pred, label):
        cm = confusion_matrix(y_true, y_pred)
        return cm[self.digits.index(label), self.digits.index(label)] \
            / np.sum(cm[:, self.digits.index(label)])


    def recall_all(self, y_true, y_pred, label):
        cm = confusion_matrix(y_true, y_pred)
        return cm[self.digits.index(label), self.digits.index(label)] /\
             np.sum(cm[self.digits.index(label), :])


    def f1_score_all(self, y_true, y_pred, label):
        return 2 * self.precision_all(y_true, y_pred, label) * self.recall_all(y_true, y_pred, label)\
            / (self.precision_all(y_true, y_pred, label) + self.recall_all(y_true, y_pred, label))


    def weighted_f1_score_all(self, y_true, y_pred):
        return sum([self.f1_score_all(y_true, y_pred, label) for label in self.digits]) / len(self.digits)
    
    def plot_cm(self, y_true, y_pred):
        confusion_matrix_plot(y_true, y_pred)
        plt.show()

    def print_metrics_all(self, y_true, y_pred):

        labels = self.digits
        n = len(labels)

        print("Relatório de Classificação")
        print("\n")
        print(f"Acurácia: {self.acurracy_all(y_true, y_pred):.2f}")
        print(f"Erro: {self.error_all(y_true, y_pred):.2f}")
        for d in self.digits:
            print("\n")
            print(f"Precisão dígito {d}: {self.precision_all(y_true, y_pred, d):.2f}")
            print(f"Recall dígito {d}: {self.recall_all(y_true, y_pred, d):.2f}")
            print(f"F1 Score dígito {d}: {self.f1_score_all(y_true, y_pred, d):.2f}")
        print("\n")
        print(f"Weighted F1 Score: {self.weighted_f1_score_all(y_true, y_pred):.2f}")
        print("\n")