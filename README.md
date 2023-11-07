# two-step_dopamine

![GitHub_fig](https://github.com/Mblancopozo/two-step_dopamine/assets/34422444/cddb99db-bba6-48c7-bb63-11274612d545)

This repository contains code used to generate the main figures from the [manuscript](https://www.biorxiv.org/content/10.1101/2021.06.25.449995v3):

> Blanco-Pozo, M., Akam, T., &  Walton, M. (2023).  **Dopamine-independent state inference mediates expert reward guided decision making**  *bioRxiv*, 2023-04.

## Usage:

The file [main_script.py](./main_script.py) contains the functions to import the data and generate the main figures.
[main_script.py](./main_script.py) is divided into several sections: 1) Imports, 2) Import manuscript data, 3) Time-wrap trials, 4) Dictionary structure for analysis, 5) Import saved variables, 6) Figure 1C-F, 7) Figure 1G-J, 8) Figure 2 & 4F - dopamine z-score plots, 9) Figure 3, 4, S4-8 - Photometry regression, 10) Figure 5 - Optogenetics

The `raw_data` folder contains the behavioural and photometry files
The `data_variables` folder contains the imported and preprocessed data. 
The `code` folder contains scripts with functions for analysis and plotting. These functions are called from [main_script.py](./main_script.py).

Instead of importing and preprocessing the data from the behavioural and photometry files in the `raw_data` folder, the imported and preprocessed variables can be found in the `data_variables` folder. Then, from the [main_script.py](./main_script.py), section 5, these can be directly imported.