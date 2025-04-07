import os
import numpy as np
from cmdstanpy import CmdStanModel
from typing import Any, Dict

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
