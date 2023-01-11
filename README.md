# Cautious explorers generate more future academic impact
This repository contains the processed data (`data` directory) and replication code (`code` directory) for the paper 'Cautious explorers generate more future academic impact'. 

## `data` directory
The academic publication data used in the paper is provided by [APS](https://journals.aps.org/datasets) and [MAG](https://learn.microsoft.com/en-us/academic-services/graph/), and the source data is available on request from the corresponding official websites. 
This directory contains the processed data for subsequent analyses and experiments.

## `code` directory
- `1_processed_data.ipynb` is responsible for the processing and statistical description of the raw data;
- `2_interplays.ipynb` reports on the relationship between scientists' exploratory metrics (EP&ED) and performance；
- `3_sample_scientist.ipynb` visualizes sample scientists from each of the high/low EP/ED groups;
- `4_regression_result.ipynb` records the result of a regression of EP and ED on the performance of scientists；
- `5_1_PSW.R` and `5_2_PSW_result.ipynb` are the R code and results of obtaining average treatment effect (ATE) for four groups via propensity score weighting(PSW) method；
- `6_temporal_perspectives.ipynb` depicts the EP and ED in different career years (1-30) and times (1985-1989, 1990-1994, and 1995-1999), and member exchanges among the four groups in the 2000s, corresponding to Figure 5 in the main text.