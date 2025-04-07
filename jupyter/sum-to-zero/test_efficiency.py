# Compare ESS/sec using binomial model
# and simulated datasets based on smaller, larger number of observations.


import os
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple
from cmdstanpy import CmdStanModel

import logging
cmdstanpy_logger = logging.getLogger("cmdstanpy")
cmdstanpy_logger.setLevel(logging.ERROR)

import warnings
warnings.filterwarnings('ignore')

# Fit model and dataset for N iterations.
# For each run, save wall clock time and effective samples / second (N_Eff/sec)
# Return np.ndarray of size (N,2) with timing information.
def time_fits(N: int, model: CmdStanModel, data: dict) -> np.ndarray:
    fit_times = np.ndarray(shape=(N, 2), dtype=float)
    for i in range(N):
        fit = model.sample(data=data, parallel_chains=4,
                           show_progress=False, show_console=False, refresh=10_000)
        times = fit.time
        fit_summary = fit.summary()
        total_time = 0
        for j in range(len(times)):
            total_time += times[j]['total']

        fit_times[i, 0] = total_time
        fit_times[i, 1] = fit_summary.loc['lp__', 'ESS_bulk']/total_time
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
            'ESS/sec': np.mean(array, axis=0)[1]
        })
    df = pd.DataFrame(result_data)
    return df.set_index('label').round(2)    


# Runs data-generating model and returns a Dict containing simulated data and metadata.
def simulate_data(
        N_eth: int, N_edu: int, N_age: int, 
        baseline: float, sens: float, spec: float,
        N_obs_per_stratum: int, *, seed: int = 345678
        ) -> Dict[str, Any]:

    N = 2 * N_eth * N_edu * N_age * N_obs_per_stratum

    inputs_dict = {
        'N': N, 'N_eth': N_eth, 'N_edu': N_edu, 'N_age': N_age,
        'baseline': baseline, 'sens': sens, 'spec': spec
    }
    datagen_mod = CmdStanModel(stan_file=os.path.join('stan', 'gen_binomial_4_preds.stan'))
    sim_data = datagen_mod.sample(data=inputs_dict, iter_warmup=1, iter_sampling=1, chains=1,
                                       show_progress=False, show_console=False, refresh=10_000,
                                       seed=45678)
    gen_data = {
        'N':sim_data.pos_tests.shape[1], 
        'N_age':N_age,
        'N_eth':N_eth,
        'N_edu':N_edu,
        'sens': sens,
        'spec': spec,
        'intercept_prior_mean': baseline,
        'intercept_prior_scale': 2.5,
        'pos_tests':sim_data.pos_tests[0].astype(int),
        'tests':sim_data.tests[0].astype(int),
        'sex':sim_data.sex[0].astype(int),
        'age':sim_data.age[0].astype(int), 
        'eth':sim_data.eth[0].astype(int),
        'edu':sim_data.edu[0].astype(int),
        'beta_0': sim_data.beta_0[0],
        'pct_sex': sim_data.pct_sex[0],
        'beta_sex': sim_data.beta_sex[0],
        'pct_age': sim_data.pct_age[0],
        'beta_age':sim_data.beta_age[0],
        'pct_eth': sim_data.pct_eth[0],
        'beta_eth':sim_data.beta_eth[0],
        'pct_edu': sim_data.pct_edu[0],
        'beta_edu':sim_data.beta_edu[0],
        'seed':sim_data.metadata.cmdstan_config['seed'],
    }
    return gen_data


# Create a dataset - fix sizes, and seed

N_eth = 3
N_edu = 5
N_age = 9
baseline = -3.5
sens = 0.75
spec = 0.9995
data_tiny = simulate_data(N_eth, N_edu, N_age, baseline, sens, spec, 7, seed=45678)
data_small = simulate_data(N_eth, N_edu, N_age, baseline, sens, spec, 17, seed=data_tiny['seed'])
data_large = simulate_data(N_eth, N_edu, N_age, baseline, sens, spec, 200, seed=data_tiny['seed'])


# sum to zero vector

binomial_ozs_mod = CmdStanModel(stan_file=os.path.join('stan', 'binomial_4preds_ozs.stan'))
times_ozs_large = time_fits(100, binomial_ozs_mod, data_large)
times_ozs_small = time_fits(100, binomial_ozs_mod, data_small)

# sum to zero vector

binomial_hard_mod = CmdStanModel(stan_file=os.path.join('stan', 'binomial_4preds_hard.stan'))
times_hard_small = time_fits(100, binomial_hard_mod, data_small)
times_hard_large = time_fits(100, binomial_hard_mod, data_large)


# soft sum-to-zero constraint

binomial_soft_mod = CmdStanModel(stan_file=os.path.join('stan', 'binomial_4preds_soft.stan'))
times_soft_small = time_fits(100, binomial_soft_mod, data_small)
times_soft_large = time_fits(100, binomial_soft_mod, data_large)


df_summary = summarize_times([('ozs small', times_ozs_small),
                              ('ozs large', times_ozs_large),
                              ('hard small', times_hard_small),
                              ('hard large', times_hard_large),
                              ('soft small', times_soft_small),
                              ('soft large', times_soft_large)])
df_summary.to_json("binomial_runtimes.json")

