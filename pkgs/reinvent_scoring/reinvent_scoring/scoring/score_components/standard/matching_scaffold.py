from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import rdScaffoldNetwork
from rdkit.Chem.Scaffolds import MurckoScaffold

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.score_components import BaseScoreComponent
from reinvent_scoring.scoring.score_summary import ComponentSummary


class MatchingScaffold(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        self.target_smarts = self.parameters.specific_parameters.get(self.component_specific_parameters.SMILES, [])
        self._validate_inputs(self.target_smarts)
        params = rdScaffoldNetwork.ScaffoldNetworkParams()
        params.includeScaffoldsWithAttachments=False
        params.includeScaffoldsWithoutAttachments=True
        params.collectMolCounts=False
        params.includeGenericScaffolds=False
        self.params = params

    def calculate_score(self, molecules: List) -> ComponentSummary:
        score = self._substructure_match(molecules, self.target_smarts)
        score_summary = ComponentSummary(total_score=score, parameters=self.parameters)
        return score_summary

    def _substructure_match(self, query_mols, list_of_SMILES):
        if len(list_of_SMILES) == 0:
            return np.ones(len(query_mols), dtype=np.float32)
        list_of_SMILES = [MurckoScaffold.MurckoScaffoldSmiles(subst) for subst in list_of_SMILES]

        match = []
        for mol in query_mols:
            try:
                net = rdScaffoldNetwork.CreateScaffoldNetwork([mol], self.params)
            except:
                net = []
            scaffolds = net.nodes
            result = 0
            for subst in list_of_SMILES:
                if Chem.MolToSmiles(mol) == subst:
                    break
                for scaffold in scaffolds:
                    if subst == scaffold:
                        result = 1
                        break
                if result == 1:
                    break

            match.append(result)

        return np.array(match)

    def _validate_inputs(self, smiles):
        for smart in smiles:
            if Chem.MolFromSmarts(smart) is None:
                raise IOError(f"Invalid smarts pattern provided as a matching substructure: {smart}")
