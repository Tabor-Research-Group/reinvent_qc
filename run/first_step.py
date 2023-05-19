# load dependencies
import os
import re
import json
import tempfile

# --------- change these path variables as required
reinvent_dir = os.path.expanduser("../Reinvent")
reinvent_env = os.path.expanduser(".conda/envs/reinvent")
output_dir = os.path.expanduser("~/first_step_demo")

# --------- do not change
# get the notebook's root path
try: ipynb_path
except NameError: ipynb_path = os.getcwd()

# if required, generate a folder to store the results
try:
    os.mkdir(output_dir)
except FileExistsError:
    pass

# initialize the dictionary
configuration = {
    "version": 3.2,                        # we are going to use REINVENT's newest release
    "run_type": "reinforcement_learning"   # other run types: "sampling", "validation",
                                           #                  "transfer_learning",
                                           #                  "scoring" and "create_model"
}

# add block to specify whether to run locally or not and
# where to store the results and logging
configuration["logging"] = {
    "sender": "http://0.0.0.1",          # only relevant if "recipient" is set to "remote"
    "recipient": "local",                # either to local logging or use a remote REST-interface
    "logging_frequency": 10,             # log every x-th steps
    "logging_path": os.path.join(output_dir, "progress.log"), # load this folder in tensorboard
    "result_folder": os.path.join(output_dir, "results"),     # will hold the compounds (SMILES) and summaries
    "job_name": "Reinforcement learning demo",                # set an arbitrary job name for identification
    "job_id": "demo"                     # only relevant if "recipient" is set to a specific REST endpoint
}

# add the "parameters" block
configuration["parameters"] = {}

# add a "diversity_filter"
configuration["parameters"]["diversity_filter"] =  {
    "name": "IdenticalMurckoScaffold",     # other options are: "IdenticalTopologicalScaffold", 
                                           # "IdenticalMurckoScaffold" and "ScaffoldSimilarity"
                                           # -> use "NoFilter" to disable this feature
    "nbmax": 16,                           # the bin size; penalization will start once this is exceeded
    "minscore": 0.85,                      # the minimum total score to be considered for binning
    "minsimilarity": 0.6                   # the minimum similarity to be placed into the same bin
}

# prepare the inception (we do not use it in this example, so "smiles" is an empty list)
configuration["parameters"]["inception"] = {
    "smiles": [],                         # fill in a list of SMILES here that can be used (or leave empty)
    "memory_size": 0,                     # sets how many molecules are to be remembered
    "sample_size": 0                      # how many are to be sampled each epoch from the memory
}

# set all "reinforcement learning"-specific run parameters
configuration["parameters"]["reinforcement_learning"] = {
    "prior": os.path.join(ipynb_path, "../data/random.prior.new"), # path to the pre-trained model
    "agent": os.path.join(ipynb_path, "../data/random.prior.new"), # path to the pre-trained model
    "n_steps": 1000,                       # the number of epochs (steps) to be performed; often 1000
    "sigma": 128,                          # used to calculate the "augmented likelihood", see publication
    "learning_rate": 0.0001,               # sets how strongly the agent is influenced by each epoch
    "batch_size": 128,                     # specifies how many molecules are generated per epoch
    "margin_threshold": 50                 # specify the (positive) margin between agent and prior
}

# prepare the scoring function definition and add at the end
scoring_function = {
    "name": "custom_product",              # this is our default one (alternative: "custom_sum")
    "parallel": False,                     # sets whether components are to be executed
                                           # in parallel; note, that python uses "False" / "True"
                                           # but the JSON "false" / "true"

    # the "parameters" list holds the individual components
    "parameters": [

    # add component: consecutive rotatable bond
    {
        "component_type": "cons_rotatable_bonds",  # detection of existance of any consecutive rotatable bond 
        "name": "Consecutive rotatable bonds",     # arbitrary name for the component
        "weight": 1                                # the weight ("importance") of the component (default: 1)                      
    },

    # add component: molecular weight 
    {
        "component_type": "molecular_weight", 
        "name": "Molecular weight",           # arbitrary name for the component
        "weight": 1,                          # the weight ("importance") of the component (default: 1)                      
        "specific_parameters": {
            "transformation": {
                "transformation_type": "left_step",     # left step transformation 
                "low": 400,
            }
        }
    },

    # add component: enforce to NOT match a given substructure
    {
        "component_type": "custom_alerts",
        "name": "Custom alerts",               # arbitrary name for the component
        "weight": 1,                           # the weight of the component (default: 1)
        "specific_parameters": {
            "smiles": [                        # specify the substructures (as list) to penalize
                "[*;r3]",
                "[*;r4]",
                "[*;r7]",
                "[*;r8]",
                "[*;r9]",
                "[*;r10]",
                "[*;r11]",
                "[*;r12]",
                "[*;r13]",
                "[*;r14]",
                "[*;r15]",
                "[*;r16]",
                "[*;r17]",
                "[#7]~[#7]",
                "N=c1[nH]cccc1",
                "N=c1ccc[nH]1",
                "c1onccc1",
                "[#8]~[#8]",
                "[#7]~[#8]",
                "[#6;+]",
                "[#16][#16]",
                "[#7;!n][S;!$(S(=O)=O)]",
                "[#7;!n][#7;!n]",
                "[#7;!n][C;!$(C(=[O,N])[N,O])][#16;!s]",
                "[#7;!n][C;!$(C(=[O,N])[N,O])][#7;!n]",
                "[#7;!n][C;!$(C(=[O,N])[N,O])][#8;!o]",
                "[#8;!o][C;!$(C(=[O,N])[N,O])][#16;!s]",
                "[#8;!o][C;!$(C(=[O,N])[N,O])][#8;!o]",
                "[#16;!s][C;!$(C(=[O,N])[N,O])][#16;!s]",
                "C=c1ccccc1=C",
                "N=c1cc-ccc1",
                "Cn1cccc1=C",
                "Cn1ccc(=C)c1",
                "n1cccc1=C",
                "n1ccc(=C)c1",
                "C1C=c2ccccc2=N1",
                "c1sc(=N)[nH]c1",
                "c1[nH]c(=O)ccc1",
                "N=c1nccc[nH]1",
                "N=c1ncc[nH]1",
                "[*]=N~[*]",
                "[*;+]",
                "[*;-]",
                "[*]~1~[*]~[*]=[#6]=[*]~[*]~1",
                "[*]~1~[*]=[#6]=[*]~[*]~1"
            ]
        }
    
    },
    
    # add component: SCScore
    {
        "component_type": "scscore",  # SCScore for synthesizability estimation 
        "name": "SCScore",            # arbitrary name for the component
        "weight": 1,                  # the weight ("importance") of the component (default: 1)                      
        "specific_parameters": {
            "transformation": {
                "transformation_type": "left_step", # left step transformation 
                "low": 4,
            }
        }
    },

    ]
}
configuration["parameters"]["scoring_function"] = scoring_function

# write the configuration file to the disc
configuration_JSON_path = os.path.join(output_dir, "job_config.json")
with open(configuration_JSON_path, 'w') as f:
    json.dump(configuration, f, indent=4, sort_keys=True)

# execute REINVENT from the command-line
os.system(f'{reinvent_env}/bin/python {reinvent_dir}/input.py {configuration_JSON_path}')
