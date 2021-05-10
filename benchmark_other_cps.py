from multiprocessing import Pool, cpu_count

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

# for performance reasons, we run the datasets in parallel
# this means every dataset will have its own results folder
benchmarkName = "cps"
all_datasets = [
    '10rooms',
    'aircraft',
    'cartpole',
    'dcdc',
    'helicopter',
    'traffic_30m',
    'truck_trailer',
]
NUM_PROCESSES = min(cpu_count()-1, len(all_datasets))

def runBenchmark(dataset):
    baseFolder = f"results/{benchmarkName}/{dataset}"
    suite = BenchmarkSuite(timeout=60*60*3,
                           save_folder=f"{baseFolder}/saved_classifiers",
                           output_folder=f"{baseFolder}/generated_trees",
                           benchmark_file=baseFolder,
                           rerun=False)

    suite.add_datasets(['controllers_cps'], include=[dataset])

    aa = AxisAlignedSplittingStrategy()
    lin_logreg = LinearClassifierSplittingStrategy(LogisticRegression, solver='lbfgs', penalty='none')
    lin_svm = LinearClassifierSplittingStrategy(LinearSVC, max_iter=5000, dual=False)
    lin_oc1 = OC1SplittingStrategy()

    poly = PolynomialClassifierSplittingStrategy(prettify=False)
    poly.priority = 0.1
    polyPrio1 = PolynomialClassifierSplittingStrategy(prettify=False)
    polyPrio1.priority = 1.0

    entropy = Entropy(determinizer=LabelPowersetDeterminizer())
    minEntropy = MinLabelEntropy(determinizer=LabelPowersetDeterminizer())

    classifiers = [
        DecisionTree([aa],              entropy,    'axis-aligned'),
        DecisionTree([aa, lin_logreg],  entropy,    'lin-logreg'),
        DecisionTree([aa, lin_svm],     entropy,    'lin-svm'),
        DecisionTree([aa, lin_oc1],     entropy,    'lin-oc1'),
        DecisionTree([aa, poly],        entropy,    'poly'),
        DecisionTree([aa, polyPrio1],   entropy,    'polyPrio1'),

        DecisionTree([aa],              minEntropy, 'axis-aligned-minEntropy'),
        DecisionTree([aa, lin_logreg],  minEntropy, 'lin-logreg-minEntropy'),
        DecisionTree([aa, lin_svm],     minEntropy, 'lin-svm-minEntropy'),
        DecisionTree([aa, lin_oc1],     minEntropy, 'lin-oc1-minEntropy'),
        DecisionTree([aa, poly],        minEntropy, 'poly-minEntropy'),
        DecisionTree([aa, polyPrio1],   minEntropy, 'polyPrio1-minEntropy'),
    ]
    suite.benchmark(classifiers)
    # suite.display_html()



print(f"Running benchmark with {NUM_PROCESSES} processes..")
with Pool(processes=NUM_PROCESSES) as p:
    p.map(runBenchmark, [dataset for dataset in all_datasets])
    print("+-----------------------------------------------------------+")
    print("|                   ALL BENCHMARKS FINISHED                 |")
    print("+-----------------------------------------------------------+")