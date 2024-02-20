import re


class Tokenizer:
    """An example tokenizer for SMILES strings."""

    def __init__(self, start_token: str = "GO", end_token: str = "EOS"):
        self.start_token = start_token
        self.end_token = end_token

    @staticmethod
    def replace_halogen(string: str) -> str:
        """Regex to replace Br and Cl with single letters."""
        br = re.compile("Br")
        cl = re.compile("Cl")
        string = br.sub("R", string)
        string = cl.sub("L", string)
        return string

    def tokenize(self, smiles: str) -> list[str]:
        regex = "(\[[^\[\]]{1,6}\])"
        smiles = self.replace_halogen(smiles)
        char_list = re.split(regex, smiles)
        tokenized = [self.start_token]
        for char in char_list:
            if char.startswith("["):
                tokenized.append(char)
            else:
                [tokenized.append(unit) for unit in list(char)]
        tokenized.append(self.end_token)
        tokenized = [s.replace("L", "Cl").replace("R", "Br") for s in tokenized]
        return tokenized

    def detokenize(self, tokenized: list[str]) -> str:
        raise NotImplementedError
