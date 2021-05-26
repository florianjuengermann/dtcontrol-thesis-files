# dtcontrol Thesis Files

This repository contains files used in the bachelor's thesis _Learning Algebraic Predicates for Explainable Controllers_.
Most importantly, the 5 benchmark files can be used to reproduce the tables and charts from Chapter 7.


The repository has the following structure:
- `controllers_cps` and `controllers_qv` contain controllers for cyber-physical systems case studies and case studies from the quantitative verification benchmark set. Before using them in `dtcontrol`, you need to unzip them (e.g. by running `unzip "*.zip"` in the respective folder).
- `domain_knowledge` contains predicate sets that were generated from domain knowledge (for the `cruise` case study).
- `models` contain model files for the `cruise` example which can be used to synthesize safe controllers. See the respective readme for more information.
- `predicate_generation` contains a python script used to generate the predicates sets in `domain_knowledge`.
- `results` contains the `.json` files generated by running the benchmarks.