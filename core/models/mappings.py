from typing import Union

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from catboost import CatBoostClassifier
from entities import TrainingParams

metrics = {
    'accuracy': accuracy_score,
    'recall': recall_score,
    'f1': f1_score,
    'roc_auc': roc_auc_score
}


def get_models(train_params: TrainingParams):
    seed = train_params.random_state
    classifiers = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=seed),
        'Linear SVM': SVC(C=0.025, kernel='linear', random_state=seed),
        'RBF SVM': SVC(C=1, gamma=2, random_state=seed),
        'Gaussian Process':  GaussianProcessClassifier(1.0 * RBF(1.0), random_state=seed),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=seed),
        'Random Forest': RandomForestClassifier(max_depth=5, max_features=1, n_estimators=10, random_state=seed),
        'Neural Net': MLPClassifier(alpha=1, max_iter=1000, random_state=seed),
        'AdaBoost': AdaBoostClassifier(random_state=seed),
        'Naive Bayes': GaussianNB(),
        'QDA': QuadraticDiscriminantAnalysis(),
        'CatBoost': CatBoostClassifier(random_seed=seed)
    }
    return classifiers


ClassifierModelType = Union[
    LogisticRegression,
    SVC,
    GaussianProcessClassifier,
    DecisionTreeClassifier,
    RandomForestClassifier,
    MLPClassifier,
    AdaBoostClassifier,
    GaussianNB,
    QuadraticDiscriminantAnalysis,
    CatBoostClassifier
]

