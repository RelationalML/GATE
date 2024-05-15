# GATE: How to Keep Out Intrusive Neighbors (ICML 2024)

Experimental code for the paper 'GATE: How to Keep Out Itrusive Neighbors' to be published at ICML 2024

For synthetic data generation, specify parameter settings in 'SyntheticDataSettings.csv' and run 'generateSyntheticData.py' after setting corresponding 'graphID' in the script and calling the appropriate function. The resulting dataset 'D_<graphID>'.pkl is stored in directory 'SyntheticData/'.

Experiment settings are not input using command-line arguments but specified in a 'ExpSettings_*.csv' file and the corresponding IDs are set in the 'runExp_*.py' before running the script. Exp results are stored as .pkl files in the directory 'ExpResults_*/' for further analysis.

Cora, Citeseer and OGB datasets will be download on first use. Heterophilic datasets should be downloaded from the following https://github.com/yandex-research/heterophilous-graphs/tree/main/data into the directory 'data/heterophilous-graphs/'.


