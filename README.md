# CARSM
Code for Critic-ARSM policy gradient algorithm

This Github repo contains codes for the experiments of **Discrete Action On-Policy Learning with Action-Value Critic** paper

There are six .py files in total, and one baselines folder includes dependencies we need to run experiment. 

Among those six files,

toy.py contains code for running experiment of Toy Example.

carsm_util.py, carsm_util_2.py are files with utility functions for running carsm_trpo.py and carsm.py.
trpo_agent.py contains code for running carsm_trpo.py.

carsm.py contains code for running CARSM algorithm on a default swimmer task.

carsm_trpo.py contains code for running CARSM+TRPO algorithm on a default swimmer task.
