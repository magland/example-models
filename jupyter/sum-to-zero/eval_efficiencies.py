# Compare ESS/sec using binomial model
# and simulated datasets based on smaller, larger number of observations.
import os
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple
from cmdstanpy import CmdStanModel, set_cmdstan_path


import logging
cmdstanpy_logger = logging.getLogger("cmdstanpy")
cmdstanpy_logger.setLevel(logging.FATAL)

import warnings
warnings.filterwarnings('ignore')

set_cmdstan_path(os.path.join('/Users', 'mitzi', 'github', 'stan-dev', 'cmdstan'))

from utils import simulate_data

# Fit model and dataset for N iterations.
# For each run, save wall clock time and effective samples / second (N_Eff/sec)
# Return np.ndarray of size (N,2) with timing information.
def time_fits(N: int, model: CmdStanModel, data: dict) -> np.ndarray:
    fit_times = np.ndarray(shape=(N, 2), dtype=float)
    for i in range(N):
        print('Run', i)
        fit = model.sample(data=data, parallel_chains=4,
                           show_progress=False, show_console=False, refresh=10_000)
        fit_summary = fit.summary()
        total_time = 0
        times = fit.time
        for j in range(len(times)):
            total_time += times[j]['total']

        fit_times[i, 0] = total_time
        fit_times[i, 1] = fit_summary.loc['lp__', 'ESS_bulk/s']
    return fit_times


# Given a list of label, time pairs, populate dataframe
# of means of time and std dev wall clock time, and N_Eff/sec
def summarize_times(data_pairs: List[Tuple[str, np.ndarray]]) -> pd.DataFrame:
    result_data = []
    for label, array in data_pairs:
        result_data.append({
            'label': label,
            'mean': np.mean(array, axis=0)[0],
            'std dev': np.std(array, axis=0)[0],
            'ESS_bulk/s': np.mean(array, axis=0)[1]
        })
    df = pd.DataFrame(result_data)
    return df.set_index('label').round(2)    


# Create datasets - fix sizes, and seed
N_eth = 3
N_edu = 5
N_age = 9
baseline = -3.5
sens = 0.75
spec = 0.9995
data_small = simulate_data(N_eth, N_edu, N_age, baseline, sens, spec, 17, seed=45678)
data_large = simulate_data(N_eth, N_edu, N_age, baseline, sens, spec, 200, seed=45678)

# sum to zero vector

binomial_ozs_mod = CmdStanModel(stan_file=os.path.join('stan', 'binomial_4preds_ozs.stan'))
times_ozs_large = time_fits(100, binomial_ozs_mod, data_large)
times_ozs_small = time_fits(100, binomial_ozs_mod, data_small)

# hard sum-to-zero constraint

binomial_hard_mod = CmdStanModel(stan_file=os.path.join('stan', 'binomial_4preds_hard.stan'))
times_hard_small = time_fits(100, binomial_hard_mod, data_small)
times_hard_large = time_fits(100, binomial_hard_mod, data_large)

# soft sum-to-zero constraint

binomial_soft_mod = CmdStanModel(stan_file=os.path.join('stan', 'binomial_4preds_soft.stan'))
times_soft_small = time_fits(100, binomial_soft_mod, data_small)
times_soft_large = time_fits(100, binomial_soft_mod, data_large)


df_small = summarize_times([('ozs small', times_ozs_small),
                                ('hard small', times_hard_small),
                                ('soft small', times_soft_small)])
df_small.to_json("binomial_runtimes_small.json")

df_large = summarize_times([('ozs large', times_ozs_large),
                                ('hard large', times_hard_large),
                                ('soft large', times_soft_large)])
df_large.to_json("binomial_runtimes_large.json")
