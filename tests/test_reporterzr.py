import pytest
from sklearn import datasets
from sklearn.svm import SVC

from classification_reportzr.reporterzr import Reporterzr

def test():
    digits = datasets.load_digits()
    samples, labels = digits.data[:-1], digits.target[:-1]

    svc_kwargs = {'C':100.0, 'gamma':0.001}
    svc_reporter = Reporterzr(SVC, svc_kwargs, samples, labels)

    svc_reporter.run_experiment()
    svc_reporter.present_all_classification_report()
