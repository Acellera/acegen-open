from acegen.vocabulary.base import Tokenizer, Vocabulary
from acegen.vocabulary.tokenizers import (
    AISTokenizer,
    DeepSMILESTokenizer,
    SAFETokenizer,
    SELFIESTokenizer,
    SMILESTokenizer,
    SMILESTokenizer2,
)
from acegen.vocabulary.vocabulary import SMILESVocabulary

tokenizer_options = {
    "AISTokenizer": AISTokenizer,
    "DeepSMILESTokenizer": DeepSMILESTokenizer,
    "SAFETokenizer": SAFETokenizer,
    "SELFIESTokenizer": SELFIESTokenizer,
    "SMILESTokenizer": SMILESTokenizer,
    "SMILESTokenizer2": SMILESTokenizer2,
}