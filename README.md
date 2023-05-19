
# REINVENT with Quantum Chemistry Calculations

This repository includes:

 - REINVENT and related packages in `./REINVENT` and `./pkgs`
 - Initial ChEMBL prior network in `./data`
 - Results for generated molecules in `./results`
 - Scripts for setting up configurations and starting training for REINVENT in `./run`

## Installation
> Note: The part of source codes related to job management and data parsing may need to be modified to fit your own system.

 1. Clone this GitHub repo
 2. Create a conda environment with reinvent.yml
```bash
$ conda env create -f reinvent.yml
```
## Requirements

 - [SCScore](https://github.com/connorcoley/scscore): Synthetic complexity score prediction model
 - [AiZynthFinder](https://github.com/MolecularAI/aizynthfinder): Retrosynthetic planning tool

## Acknowledgments

 - [REINVENT 3.2](https://github.com/MolecularAI/Reinvent/tree/master): REINVENT model
 - [ReinventCommunity](https://github.com/MolecularAI/ReinventCommunity): Useful tutorials for REINVENT in jupyter notebooks
