import json
import os
import shutil
import subprocess
from sys import stdout
import tempfile
import time
import django
import datetime
import copy

from reinvent_scoring.scoring.score_summary import ComponentSummary

os.environ["DJANGO_SETTINGS_MODULE"] = "djangochem.settings.default"
django.setup()
from pgmols.models import Calc
from jobs.models import Job

from django.core.management import call_command

import numpy as np
from typing import List, Tuple

from reinvent_scoring.scoring.utils import _is_development_environment

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.score_components.console_invoked.base_console_invoked_component import BaseConsoleInvokedComponent


class RunJobs(BaseConsoleInvokedComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        self._project_name = self.parameters.specific_parameters[self.component_specific_parameters.JOB_PROJECT_NAME]
        self._tag = self.parameters.specific_parameters[self.component_specific_parameters.JOB_TAG]
        self._chemconfig = self.parameters.specific_parameters[self.component_specific_parameters.JOB_CHEMCONFIG]
        self._job_name = self.parameters.specific_parameters[self.component_specific_parameters.JOB_JOB_NAME]
        self._jobbuild_dir = self.parameters.specific_parameters[self.component_specific_parameters.JOB_JOBBUILD_DIR]
        self._jobparse_dir = self.parameters.specific_parameters[self.component_specific_parameters.JOB_JOBPARSE_DIR]
        self._target = self.parameters.specific_parameters[self.component_specific_parameters.JOB_TARGET]

    def _addsmiles(self, smiles: List[str]):
        with open("smiles.txt", "w") as f:
            f.write("\n".join(smiles))
        with open("smiles.log", "a") as f:
            f.write("adding smiles......\n")
            f.write("\n".join(smiles))
            f.write("\n")
        infile = "smiles.txt"
        
        with open("addsmiles.txt", "a") as f:
            f.write("start to add smiles\n")
            call_command("addsmiles", self._project_name, infile, tag=[self._tag])
        
    def _requestjobs(self):
        with open("requestjobs.txt", "a") as f:
            f.write("start to request jobs\n")
            call_command('requestjobs', self._project_name, self._chemconfig, tag=[self._tag], stdout=f)

        with open("requestjobs.txt", "r") as f:   
            output = f.readlines()

        req_time = datetime.datetime.now()
        
        return req_time

    def _buildjobs(self):
        with open("buildjobs.txt", "a") as f:
            f.write("start to build jobs\n")
            call_command('buildjobs', self._project_name, self._jobbuild_dir, config=self._chemconfig, batchsize=500, stdout=f)
        
        with open("buildjobs.txt", "r") as f:   
            output = f.readlines()

    def _runjobs(self):
        cwd = os.getcwd()
        os.chdir(self._jobbuild_dir)
        jobid_list = list()
        command = f"squeue -u $USER | grep {self._job_name} | awk '{{print $1}}'"
        init_set = set(subprocess.check_output(command, shell=True).decode().split())
        for iter in os.scandir('./'):
            if iter.is_dir():
                os.chdir(iter.path)
                os.system('sbatch job_grace.sh')
                tmp_set = copy.deepcopy(init_set)
                while not (tmp_set - init_set):
                    tmp_set = set(subprocess.check_output(command, shell=True).decode().split())                
                    try:
                        jobid = list(tmp_set - init_set)[0]
                        jobid_list.append(jobid)
                    except:
                        pass
                    time.sleep(1)
                os.chdir('..')
        
        os.chdir(cwd)

        tmp_list = copy.deepcopy(jobid_list)
        while len(tmp_list) > 0:
            for idx in range(len(jobid_list)):
                jobid = jobid_list[idx]
                # check if exit code is 0, if not, job is finished and should be removed from the list
                if os.system(f"squeue -u $USER | grep {jobid}") != 0: 
                    try:
                        tmp_list.remove(jobid)
                    except:
                        pass

            time.sleep(60)
        
    def _parsejobs(self, smiles: List[str], req_time):
        with open("parsejobs.txt", "a") as f:
            call_command('parsejobs', self._project_name, self._jobbuild_dir, root_path=self._jobparse_dir, stdout=f)

        name_list = [idx for idx in range(len(smiles))]
        value_list = list()

        job = Job.objects.filter(group__name=self._project_name, status='done', config__name=self._chemconfig, createtime__lte=req_time)
        for smi in smiles:
            calc = Calc.objects.filter(mol__smiles=smi, mol__tags__contains=[self._tag], parentjob__in=job)
            if len(calc) != 0:
                value_list.append(calc[0].props[self._target])
            else:
                value_list.append(None)

        cwd = os.getcwd()
        os.chdir(self._jobbuild_dir)
        # move all the folders and files in the current folder to the error folder
        for iter in os.scandir('./'):
            if iter.is_dir():
                os.system(f"mv {iter.path} ../error")

        os.chdir(cwd)

        return name_list, value_list

    def _calculate_score(self, smiles: List[str], step) -> np.array:
        # add the SMILES to the database
        self._addsmiles(smiles)

        # request jobs from the database
        req_time = self._requestjobs()

        # build the jobs
        self._buildjobs()

        # run the jobs
        self._runjobs()

        # parse the jobs
        smiles_ids, scores = self._parsejobs(smiles=smiles, req_time=req_time)

        # apply transformation
        transform_params = self.parameters.specific_parameters.get(
            self.component_specific_parameters.TRANSFORMATION, {}
        )
        transformed_scores = self._transformation_function(scores, transform_params)

        return np.array(transformed_scores), np.array(scores)
