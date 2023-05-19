import numpy as np
import pandas as pd
from typing import Tuple, List

from running_modes.configurations.reinforcement_learning.inception_configuration import InceptionConfiguration
from reinvent_chemistry.conversions import Conversions
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


class Inception:
    def __init__(self, configuration: InceptionConfiguration, scoring_function, prior):
        self.configuration = configuration
        self._chemistry = Conversions()
        self.memory: pd.DataFrame = pd.DataFrame(columns=['smiles', 'scaffolds', 'score', 'likelihood'])
        self._load_to_memory(scoring_function, prior, self.configuration.smiles)


    def _load_to_memory(self, scoring_function, prior, smiles):
        if len(smiles):
            standardized_and_nulls = [self._chemistry.convert_to_rdkit_smiles(smile) for smile in smiles]
            standardized = [smile for smile in standardized_and_nulls if smile is not None]
            self.evaluate_and_add(standardized, scoring_function, prior)

    def _purge_memory(self):
        sorted_df = self.memory.sort_values('score', ascending=False).dropna()
        grouped_df = sorted_df.groupby('scaffolds').head(10)
        self.memory = grouped_df.head(self.configuration.memory_size)

    def evaluate_and_add(self, smiles, scoring_function, prior):
        if len(smiles) > 0:
            mols = [Chem.MolFromSmiles(smi) for smi in smiles]
            scaffold_mols = [MurckoScaffold.MakeScaffoldGeneric(MurckoScaffold.GetScaffoldForMol(mol)) for mol in mols]
            scaffolds = [Chem.MolToSmiles(scaffold_mol, isomericSmiles=False) for scaffold_mol in scaffold_mols]
            score = scoring_function.get_final_score(smiles)
            likelihood = prior.likelihood_smiles(smiles)
            df = pd.DataFrame({"smiles": smiles, "scaffolds": scaffolds, "score": score.total_score, "likelihood": -likelihood.detach().cpu().numpy()})
            self.memory = self.memory.append(df)
            self._purge_memory()

    def diversity_filter(self, filter_memory, bucket_size, minscore):
        if len(self.memory) > 0:
            df = self.memory.copy()
            for scaffold in set(filter_memory.Scaffold.values):
                if (filter_memory["Scaffold"].values == scaffold).sum() > bucket_size:
                    df = df[~df.scaffolds.isin(filter_memory.Scaffold)].copy() 
            self.memory = df


    def add(self, smiles, score, neg_likelihood):
        # NOTE: likelihood should be already negative
        if len(smiles) > 0:
            scaffolds = list()
            for smi in smiles:
                try:
                    mol = Chem.MolFromSmiles(smi)
                    scaffold_mol = MurckoScaffold.MakeScaffoldGeneric(MurckoScaffold.GetScaffoldForMol(mol))
                    scaffolds.append(Chem.MolToSmiles(scaffold_mol, isomericSmiles=False))
                except:
                    scaffolds.append(None)
            df = pd.DataFrame({"smiles": smiles, "scaffolds": scaffolds, "score": score, "likelihood": neg_likelihood.detach().cpu().numpy()})
            self.memory = self.memory.append(df)
            self._purge_memory()

    def sample(self) -> Tuple[List[str], np.array, np.array]:
        sample_size = min(len(self.memory), self.configuration.sample_size)
        if sample_size > 0:
            sampled = self.memory.sample(sample_size)
            smiles = sampled["smiles"].values
            scores = sampled["score"].values
            prior_likelihood = sampled["likelihood"].values
            return smiles, scores, prior_likelihood
        return [], [], []


