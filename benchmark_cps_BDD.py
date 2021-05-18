from dtcontrol.benchmark_suite import BenchmarkSuite
from dtcontrol.bdd import BDD

benchmarkName = "cpsBDD"
num_tries = 1
all_datasets = [
    '10rooms',
    'aircraft',
    'cartpole',
    'cruise_250',
    'cruise_300',
    'dcdc',
    'helicopter',
    'traffic_30m',
    'truck_trailer',
]

baseFolder = f"results/{benchmarkName}"
suite = BenchmarkSuite(timeout=60*60*3,
                        save_folder=f"{baseFolder}/saved_classifiers",
                        output_folder=f"{baseFolder}/generated_trees",
                        benchmark_file=baseFolder,
                        rerun=False)

suite.add_datasets(['controllers_cps'], include=all_datasets)

classifiers = [
    *[BDD(0, name_suffix=i) for i in range(num_tries)],
    *[BDD(1, name_suffix=i) for i in range(num_tries)],
]
suite.benchmark(classifiers)
# suite.display_html()
