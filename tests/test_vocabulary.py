from acegen.vocabulary.vocabulary import SMILESTokenizer, SMILESVocabulary

single_smiles = "CC1=CC=CC=C1"
multiple_smiles = [
    "CCO",  # Ethanol (C2H5OH)
    "CCN(CC)CC",  # Triethylamine (C6H15N)
    "CC(=O)OC(C)C",  # Diethyl carbonate (C7H14O3)
    "CC(C)C",  # Isobutane (C4H10)
    "CC1=CC=CC=C1",  # Toluene (C7H8)
]
chars = ["(", ")", "1", "=", "C", "N", "O"]


def test_tokenize():
    tokenizer = SMILESTokenizer()
    tokens = tokenizer.tokenize(single_smiles)
    assert sorted(tokens) == sorted(
        ["GO", "C", "C", "1", "=", "C", "C", "=", "C", "C", "=", "C", "1", "EOS"]
    )


def test_untokenize():
    tokenizer = SMILESTokenizer()
    tokens = tokenizer.tokenize(single_smiles)
    smiles = tokenizer.untokenize(tuple(tokens))
    assert smiles == single_smiles


def test_from_smiles():
    vocabulary = SMILESVocabulary.create_from_smiles(multiple_smiles)
    assert len(vocabulary) > 0


def create_from_list_of_chars():
    vocabulary = SMILESVocabulary.create_from_list_of_chars(chars)
    assert len(vocabulary) > 0


def test_create_methods_match():
    vocabulary = SMILESVocabulary.create_from_smiles(multiple_smiles)
    vocabulary2 = SMILESVocabulary.create_from_list_of_chars(sorted(chars))
    for obj1, obj2 in zip(vocabulary.__dict__.items(), vocabulary2.__dict__.items()):
        k1, v1 = obj1
        k2, v2 = obj2
        assert k1 == k2
        if k1 != "tokenizer":
            assert v1 == v2


def test_full_pipeline():
    vocabulary = SMILESVocabulary.create_from_smiles(multiple_smiles)
    for smiles in multiple_smiles:
        tokens = vocabulary.encode(smiles)
        smiles2 = vocabulary.decode(tokens)
        assert smiles == smiles2
