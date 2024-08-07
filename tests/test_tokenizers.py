import pytest

from acegen.vocabulary.tokenizers import (
    AISTokenizer,
    SAFETokenizer,
    DeepSMILESTokenizer,
    SELFIESTokenizer,
    SMILESTokenizerChEMBL,
    SMILESTokenizerEnamine,
    SMILESTokenizerGuacaMol,
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
    "CC1=CC=CC=C1",  # Toluene (C7H8)
]

@pytest.mark.parametrize("tokenizer, available, error", [
    (DeepSMILESTokenizer, _has_deepsmiles, DEEPSMILES_ERR if not _has_deepsmiles else None),
    (SELFIESTokenizer, _has_selfies, SELFIES_ERR if not _has_selfies else None),
    (SMILESTokenizerChEMBL, True, None),
    (SMILESTokenizerEnamine, True, None),
    (SMILESTokenizerGuacaMol, True, None),
    # (AISTokenizer, _has_AIS, AIS_ERR if not _has_AIS else None),
    # (SAFETokenizer, _has_SAFE, SAFE_ERR if not _has_SAFE else None),
])
def test_smiles_based_tokenizers(tokenizer, available, error):
    if not available:
        pytest.skip(f"Skipping {tokenizer.__name__} test because the required module is not available: {error}")
    for smiles in multiple_smiles:
        t = tokenizer()
        tokens = t.tokenize(smiles)
        assert len(tokens) > 0
        assert isinstance(tokens, list)
        assert isinstance(tokens[0], str)
        decoded_smiles = t.untokenize(tokens)
        assert decoded_smiles == smiles

