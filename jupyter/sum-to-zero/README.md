Case study and jupyter notebook which demonstrate the correctness and efficiency of the
`sum_to_zero_vector` constrained parameter type, introduced in Stan 2.36.


- Case study `sum_to_zero_evaluation.qmd` demonstrates use of the `sum_to_zero_vector` for 2 models.

- Jupyter notebook `sum_to_zero_evalutation.ipynb` is a step-by-step explanation of the operations
used to carry out this evaluation. 

Included in the GitHub repository for this notebook are several python files of helper functions.

* eval_efficiencies.py - run models repeatedly, get average performance stats.
* utils.py - simulate data for binomial model.
* utils\_html.py - format Stan summaries for this notebook.
* utils\_bym2.py - compute data inputs to the BYM2 model.
* utils\_nyc\_map.py - munge the New York City census tract map.

Author:  Mitzi Morris
