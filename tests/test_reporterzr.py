from sklearn import datasets
from sklearn.svm import SVC

from classification_reportzr.reporterzr import Reporterzr

def test():
    iris = datasets.load_iris()
    samples, labels = iris.data[:-1], iris.target[:-1]

    param_grid = {
        'C': [10,50,100],
        'gamma': [0.005,0.05,0.5],
        'kernel': ['poly', 'rbf', 'linear']
    }
    svc_reporter = Reporterzr(SVC, param_grid)

    test_sizes = [0.1, 0.2, 0.3]
    repetition = 7

    svc_reporter.run_experiment(samples, labels, test_sizes=test_sizes, repetition=repetition)

