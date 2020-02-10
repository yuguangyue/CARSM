# CARSM
Code for Critic-ARSM policy gradient algorithm

This Github repo contains codes for the experiments of **Discrete Action On-Policy Learning with Action-Value Critic** paper

There are six python files in total, among which, 

toy.py contains code for running experiment of Toy Example.

carsm_util.py, carsm_util_2.py are files with utility functions for running carsm_trpo.py and carsm.py.
trpo_agent.py contains code for running carsm_trpo.py.

carsm.py contains code for running CARSM algorithm on a default swimmer task.

carsm_trpo.py contains code for running CARSM+TRPO algorithm on a default swimmer task.

Code also need dependencies on openai baselines *https://github.com/openai/baselines*
