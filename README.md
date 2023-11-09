# two-step_dopamine

![GitHub_fig](https://github.com/Mblancopozo/two-step_dopamine/assets/34422444/cddb99db-bba6-48c7-bb63-11274612d545)

This repository contains code used to generate the main figures from the [manuscript](https://www.biorxiv.org/content/10.1101/2021.06.25.449995v3):

> Blanco-Pozo, M., Akam, T., &  Walton, M. (2023).  **Dopamine-independent state inference mediates expert reward guided decision making**  *bioRxiv*, 2023-04.

## Usage:

The file [main_script.py](./main_script.py) contains the functions to import the data and generate the main figures.
[main_script.py](./main_script.py) is divided into several sections: 1) Imports, 2) Import manuscript data, 3) Time-wrap trials, 4) Dictionary structure for analysis, 5) Import saved variables, 6) Figure 1C-F, 7) Figure 1G-J, 8) Figure 2 & 4F - dopamine z-score plots, 9) Figure 3, 4, S4-8 - Photometry regression, 10) Figure 5 - Optogenetics

The `analysis_code` folder contains scripts with functions for analysis and plotting. These functions are called from [main_script.py](./main_script.py).

The `RL_agents_two_step` folder contains the RL models of each class: model-free (mf), model-based (mb), Bayesian inference (latent_state), model-free with different learning rates for rewards and omissions and forgetting of the non-experienced state (mf_forget_diffa), model-based with different learning rates and forgetting parameter decaying towards 0 (mb_forget_0_diffa), and asymmetric bayesian inference (latent_state_rewasym).

The `pyControl_code` folder contains the pyControl task definition files for the Two-step task (with the different training stages), and the Two-step task for the optogenetic stimulation.

The raw data files and preprocessed variables to generate the figures can be found on [OSF](https://osf.io/u6xrc/?view_only=95ed86684f494b9582860ddcb31cd0e1)

- The `raw_data_files` folder contains the behavioural and photometry files

- The `data_variables` folder contains the imported and preprocessed data. 


Instead of importing and preprocessing the data from the behavioural and photometry files in the `raw_data` folder, the imported and preprocessed variables can be found in the `data_variables` folder. Then, from the [main_script.py](./main_script.py), section 5, these can be directly imported.
