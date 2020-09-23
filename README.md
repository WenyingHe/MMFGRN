# MMFGRN: a multi-source and multi-model fusion method for reconstruct Gene Regulatory Network

### The version of Python and packages
    Python version=3.7.6
	lightgbm version=2.3.1
	xgboost version=1.0.2
	scikit-learn version=0.22.1
	numpy version=1.18.1

### The describe of the program 

```
The program of M3_xgb.py is a single model for GRN inference on time-series and steady-state data jointly.

The program of Fusion_model.py can get the global regulatory link ranking.
```

### Parameters
	path_in_timeseries: path of time-series data
	path_in_knockouts: path of Knockout data
	path_in_knockdowns：path of Knockdown data
	path_out：the global regulatory link ranking
	samples：number of times-series experiments
	Other parameters of xgboost or lightgbm
