import pytest
import warnings
from functools import partial
from rdkit.Chem import AllChem as Chem

from acegen.vocabulary.tokenizers import (
    AISTokenizer,
    DeepSMILESTokenizer,
    SAFETokenizer,
    SELFIESTokenizer,
    SMILESTokenizerChEMBL,
    SMILESTokenizerEnamine,
    SMILESTokenizerGuacaMol,
    SmiZipTokenizer,
)

try:
    import deepsmiles

    _has_deepsmiles = True
except ImportError as err:
    _has_deepsmiles = False
    DEEPSMILES_ERR = err
try:
    import selfies

    _has_selfies = True
except ImportError as err:
    _has_selfies = False
    SELFIES_ERR = err
try:
    import smizip

    _has_smizip = True
except ImportError as err:
    _has_smizip = False
    SMIZIP_ERR = err
try:
    import atomInSmiles as AIS

    _has_AIS = True
except ImportError as err:
    _has_AIS = False
    AIS_ERR = err
try:
    import safe

    _has_SAFE = True
except ImportError as err:
    _has_SAFE = False
    SAFE_ERR = err

multiple_smiles = [
    "CCO",  # Ethanol (C2H5OH)
    "CCN(CC)CC",  # Triethylamine (C6H15N)
    "CC(=O)OC(C)C",  # Diethyl carbonate (C7H14O3)
    "CC(C)C",  # Isobutane (C4H10)
    "Cc1ccccc1",  # Toluene (C7H8)
]
ngrams = [
    "(=O)",
    "cc",
    "[C@@H]",
    "CC",
    "[C@H]",
    "(C",
    "c1ccc",
    "c2ccc",
    ")c",
    "\t",
    "\n",
    "C(=O)",
    "c1",
    "(C)",
    "c3ccc",
    "c2",
    "O)",
    "c(",
    "C(F)(F)F",
    "[nH]",
    "C(=O)N",
    "=C",
    "CCC",
    "c2ccccc2",
    "[N+](=O)[O-])",
    "(N",
    "[C@",
    "c1ccccc1",
    "c3",
    "OC",
    "(Cl)c",
    "2)",
    " ",
    "CCN",
    "COc1cc",
    "#",
    "3)",
    "%",
    "(O)",
    "NC(=O)[C@H](C",
    "(",
    ")",
    "nc",
    "+",
    ")cc1",
    "-",
    ".",
    "/",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    ":",
    "S(=O)(=O)",
    "(C)C)",
    "=",
    "c4ccc",
    "(F)c",
    "@",
    "A",
    "B",
    "C",
    "C1",
    "n1",
    "F",
    "(-",
    "H",
    "I",
    "C(=O)N[C@@H](C",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "c3ccccc3",
    "R",
    "S",
    "T",
    "CN",
    "CCNC(=N)N)",
    "c1c",
    "X",
    ")C",
    "Z",
    "[",
    "\\",
    "]",
    "O=C(",
    "CO",
    "/C=C",
    "a",
    "b",
    "c",
    "c2c",
    "e",
    "C(N)=O",
    "g",
    ")N",
    "i",
    "(OC)c",
    "C2",
    "l",
    "Cc1cc",
    "n",
    "o",
    "p",
    "[C@@]",
    "r",
    "s",
    "t",
    "n2",
    "1C",
    "=O",
    "CCN(C",
    "CC1",
    ")n",
    "C(",
    "NC(=O)",
    "[C@H](",
    "c(-",
    "[O-])",
    "2C",
    "[C@@H](",
    "C(=O)O",
    "c1n",
    ")cc",
    "cn",
    "c(N",
    "c1ccc(",
    "c1ccc2c(c1)",
    "c2n",
    "CC(C)",
    "[C@]",
    "CCCCC",
    "Cl",
    "cc2",
    "c4ccccc4",
    ")c1",
    "CCO",
    "c4",
    "(Br)c",
    "[C@H]1",
    "c(C",
    "C(C)",
    "[N+]",
    "cc1",
    "=N",
    "CN(",
    "OP(=O)(O)OC[C@H]3O[C@@H](n4cc(C)c(=O)[nH]c4=O)C[C@@H]3",
    "(C#N)",
    "c(=O)",
    "[C@@H]1",
    "C3",
    "=O)",
    "2)cc1",
    "[n+]",
    "1)",
    "C(=O)C",
    "c2ccc(",
    "N1CC",
    ")CC",
    "cc(",
    ".[Na+]",
    "(C)C",
    "c(Cl)c",
    "C)",
    "c2ccccc2)",
    "[C@H](O)",
    ")cc2)",
    "(F)",
    "[C@H]2",
    "CC(=O)N",
    "c5ccc",
    "[C@@H]2",
    "c3c",
    "[C@@H](O)",
    "C(=O)O)",
    "CCCN",
    "c(O",
    "c2c1",
    "/C=C/",
    "O=C1",
    "C=C",
    "N2CC",
    "c2ccc3c(c2)",
    "c(C(F)(F)F)c",
    "c1ccccc1)",
    "3C",
    "c3n",
    "CC2)",
    "4)",
    "N=C",
    "nn",
    "cc3",
    "[C@H](C",
    "C(C)C)",
    "c1c[nH]c2ccccc12)",
    "O=C(N",
    "C(=O)NC",
    "c(F)c",
    "C(=S)N",
    "c1ccc(Cl)cc1",
    "CC3)",
    "Cc1",
    "P(=O)(O)O",
    "N1CCN(",
    "(CC",
    "(N)",
    "c(O)c",
    "c2ccc(Cl)cc2",
    "nc2",
    "COC(=O)",
    "Cn1c",
    "Cl)",
    "c1ccc(F)cc1",
    "/C(=C",
    ")O[C@@H]3COP(=O)(S)O[C@H]3[C@@H](O)[C@H](n4c",
    "O=C(O)",
    "[C@@H](C)",
    "COc1ccc(",
    "C(N)=O)",
    "S(C)(=O)=O",
    "c(-c3cc",
    "n3",
    "OP(O)(=S)OC[C@H]",
    "c3ccccc3)",
    "[C@H](C)",
    "[C@@H]3",
    "S(=O)(=O)N",
    "nc1",
    "CS",
    "c(-c2cc",
    "CN1C(=O)",
    "c1ccc(O)cc1)",
    "C(=O)N1CCC",
    "c(C)c",
    "c3ccccc23)",
    "c2ccc(F)cc2)",
    "nc(N",
    "C[C@H](",
    "OC(C)=O)",
    "CC(",
    "CC[C@]",
    "NC(=O)[C@@H](",
    "c2ccccc12",
    "ccc1",
    "c(C(=O)N",
    "F)",
    "O=",
]
SmiZipTokenizer = partial(SmiZipTokenizer, ngrams=ngrams)
setattr(SmiZipTokenizer, "__name__", "SmiZipTokenizer")


def smiles_eq(smi1, smi2):
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    # Parse them
    if not mol1:
        return False, f"Parsing error: {smi1}"
    if not mol2:
        return False, f"Parsing error: {smi2}"
    # Remove atom map
    for mol in [mol1, mol2]:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
    # Check smiles are the same
    nsmi1 = Chem.MolToSmiles(mol1)
    nsmi2 = Chem.MolToSmiles(mol2)
    if nsmi1 != nsmi2:
        return False, f"Inequivalent SMILES: {nsmi1} vs {nsmi2}"
    # Check InChi
    inchi1 = Chem.MolToInchi(mol1)
    inchi2 = Chem.MolToInchi(mol2)
    if inchi1 != inchi2:
        return False, "Inequivalent InChi's"
    return True, ""


@pytest.mark.parametrize(
    "tokenizer, available, error",
    [
        (
            DeepSMILESTokenizer,
            _has_deepsmiles,
            DEEPSMILES_ERR if not _has_deepsmiles else None,
        ),
        (SELFIESTokenizer, _has_selfies, SELFIES_ERR if not _has_selfies else None),
        (SMILESTokenizerChEMBL, True, None),
        (SMILESTokenizerEnamine, True, None),
        (SMILESTokenizerGuacaMol, True, None),
        (AISTokenizer, _has_AIS, AIS_ERR if not _has_AIS else None),
        (SAFETokenizer, _has_SAFE, SAFE_ERR if not _has_SAFE else None),
        (SmiZipTokenizer, _has_smizip, SMIZIP_ERR if not _has_smizip else None),
    ],
)
def test_smiles_based_tokenizers(tokenizer, available, error):
    if not available:
        pytest.skip(
            f"Skipping {tokenizer.__name__} test because the required module is not available: {error}"
        )
    for smiles in multiple_smiles:
        t = tokenizer()
        tokens = t.tokenize(smiles)
        assert len(tokens) > 0
        assert isinstance(tokens, list)
        assert isinstance(tokens[0], str)
        decoded_smiles = t.untokenize(tokens)
        eq, err = smiles_eq(decoded_smiles, smiles)
        assert eq, err
        if decoded_smiles != smiles:
            warnings.warn(f"{tokenizer.__name__} behaviour: {smiles} -> {decoded_smiles}")
