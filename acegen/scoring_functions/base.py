class Task:
    """Wrapper for scoring functions that allows to limit the number of evaluations."""

    def __init__(self, scoring_function, budget):
        self.scoring_function = scoring_function
        self.budget = budget
        self.counter = 0

    @property
    def finished(self):
        return self.counter >= self.budget

    def __call__(self, smiles):
        self.counter += len(smiles)
        return self.scoring_function(smiles)
