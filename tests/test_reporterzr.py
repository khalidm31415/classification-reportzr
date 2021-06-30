from sklearn import datasets
from sklearn.svm import SVC

from classification_reportzr.reporterzr import Reporterzr

def test():
    iris = datasets.load_iris()
    samples, labels = iris.data[:-1], iris.target[:-1]

    svc_kwargs = {'C':100.0, 'gamma':0.001}
    svc_reporter = Reporterzr(SVC, svc_kwargs)

    test_sizes = [0.1, 0.2, 0.3]
    rep = 7

    svc_reporter.run_experiment(samples, labels, test_sizes=test_sizes, rep=rep)
    assert len(svc_reporter.reports) == len(test_sizes), 'The number of reports have to match the number of given test sizes'

    for report in svc_reporter.reports:
        assert report.loc['train']['accuracies'].size == rep, 'The number of measured accuracy have to match `rep`'

    svc_reporter.report_all_classification()
