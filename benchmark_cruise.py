from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from dtcontrol.benchmark_suite import BenchmarkSuite
from dtcontrol.decision_tree.decision_tree import DecisionTree
from dtcontrol.decision_tree.determinization.label_powerset_determinizer import LabelPowersetDeterminizer

from dtcontrol.decision_tree.impurity.entropy import Entropy
from dtcontrol.decision_tree.impurity.min_label_entropy import MinLabelEntropy

from dtcontrol.decision_tree.splitting.axis_aligned import AxisAlignedSplittingStrategy
from dtcontrol.decision_tree.splitting.polynomial import PolynomialClassifierSplittingStrategy
from dtcontrol.decision_tree.splitting.linear_classifier import LinearClassifierSplittingStrategy
from dtcontrol.decision_tree.splitting.oc1 import OC1SplittingStrategy


benchmarkName = "cruise"

suite = BenchmarkSuite(timeout=None,
                       save_folder=f"results/{benchmarkName}/saved_classifiers",
                       output_folder=f"results/{benchmarkName}/generated_trees",
                       benchmark_file=f"results/{benchmarkName}",
                       rerun=False)

suite.add_datasets(['controllers'], include=['cruise_250', 'cruise_300'])

aa = AxisAlignedSplittingStrategy()
lin_logreg = LinearClassifierSplittingStrategy(LogisticRegression, solver='lbfgs', penalty='none')
lin_svm = LinearClassifierSplittingStrategy(LinearSVC, max_iter=5000)
lin_oc1 = OC1SplittingStrategy()

poly = PolynomialClassifierSplittingStrategy(prettify=True)
poly.priority = 0.1

entropy = Entropy(determinizer=LabelPowersetDeterminizer())
minEntropy = MinLabelEntropy(determinizer=LabelPowersetDeterminizer())

classifiers = [
    DecisionTree([aa],              entropy,    'axis-aligned'),
    DecisionTree([aa, lin_logreg],  entropy,    'lin-logreg'),
    DecisionTree([aa, lin_svm],     entropy,    'lin-svm'),
    DecisionTree([aa, lin_oc1],     entropy,    'lin-oc1'),
    DecisionTree([aa, poly],        minEntropy, 'poly-minEntropy'),
]
suite.benchmark(classifiers)
#suite.display_html()
