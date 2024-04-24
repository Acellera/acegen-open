import os


class Task:
    """Wrapper for scoring functions that allows to limit the number of evaluations."""

    def __init__(self, scoring_function, budget, output_dir=None):
        self.scoring_function = scoring_function
        self.budget = budget
        self.counter = 0
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            self.output_file = open(output_dir + "/compounds.csv", "w")
            self.output_file.write("smiles,score\n")

    @property
    def finished(self):
        return self.counter >= self.budget

    def __call__(self, smiles):
        self.counter += len(smiles)
        scores = self.scoring_function(smiles)
        if self.output_file is not None:
            for smile, score in zip(smiles, scores):
                self.output_file.write(f"{smile},{score}\n")
        return scores
