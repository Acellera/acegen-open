import numpy as np
from reinvent_scoring.scoring import ScoringFunctionFactory
from reinvent_scoring.scoring.scoring_function_parameters import (
    ScoringFunctionParameters,
)

# Ugly hack to be able to use the scoring function
import sys
import sklearn.ensemble._forest as forest
from sklearn import tree

sys.modules["sklearn.ensemble.forest"] = forest
sys.modules["sklearn.tree.tree"] = tree


DRD2_SCORING_PARAMS = {
    "name": "custom_sum",
    "parallel": False,
    "components": [
        {
            "name": "custom_alerts",
            "component_type": "custom_alerts",
            "specific_parameters": {
                "smiles": [
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
                    "[#8][#8]",
                    "[#6;+]",
                    "[#16][#16]",
                    "[#7;!n][S;!$(S(=O)=O)]",
                    "[#7;!n][#7;!n]",
                    "C#C",
                    "C(=[O,S])[O,S]",
                    "[#7;!n][C;!$(C(=[O,N])[N,O])][#16;!s]",
                    "[#7;!n][C;!$(C(=[O,N])[N,O])][#7;!n]",
                    "[#7;!n][C;!$(C(=[O,N])[N,O])][#8;!o]",
                    "[#8;!o][C;!$(C(=[O,N])[N,O])][#16;!s]",
                    "[#8;!o][C;!$(C(=[O,N])[N,O])][#8;!o]",
                    "[#16;!s][C;!$(C(=[O,N])[N,O])][#16;!s]",
                ],
            },
            "weight": 1,
        },
        {
            "name": "DRD2",
            "component_type": "predictive_property",
            "specific_parameters": {
                "descriptor_type": "ecfp",
                "model_path": "/home/abou/torchrl_chem/scoring/drd2_ckpt.pkl",
                "radius": 3,
                "scikit": "classification",
                "size": 2048,
                "transformation": {
                    "transformation_type": "no_transformation",
                },
            },
            "weight": 1,
        },
    ],
}


class DRD2ReinventWrapper:
    def __init__(self):
        self.params = DRD2_SCORING_PARAMS

        scoring_params = ScoringFunctionParameters(
            self.params["name"], self.params["components"], self.params["parallel"]
        )

        self.scoring_class = ScoringFunctionFactory(scoring_params)

    def get_final_score(self, smiles):
        scores = self.scoring_class.get_final_score([smiles])
        output = {"valid_smile": True, "reward": float(scores.total_score[0])}
        for n, component in enumerate(self.params["components"]):
            output[component["name"]] = float(scores.profile[n].score[0])

        return output