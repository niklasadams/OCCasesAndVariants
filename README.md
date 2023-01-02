## Note
To get an updated and integrated version of process executions and variants please use [ocpa](https://gihub.com/ocpm/ocpa).



## Welcome 

### This repository provides the code to reproduce the experiments, results and figures provided in the corresponding paper.

First, unzip example_logs/csv/DS4.zip into the same directory, i.e., example_logs/csv/DS4.csv

A conda environment can be used to quickly install the dependencies:
run

``conda env create --file environment.yml``

in the anaconda prompt.

Then, run

``conda activate OC_cases_variants``

to activate the environment.

Switch to the folder of this repository and run

```python experiments.py```

to reproduce experiments and generate the figures. The generation of the isomorphism calculation times might some hours. The remaining figures should be generated within an hour each.
