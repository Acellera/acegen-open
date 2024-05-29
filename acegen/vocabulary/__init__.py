from acegen.vocabulary.base import Tokenizer, Vocabulary
from acegen.vocabulary.tokenizers import (
    AISTokenizer,
    DeepSMILESTokenizer,
    SAFETokenizer,
    SELFIESTokenizer,
    SMILESTokenizerChEMBL,
    SMILESTokenizerEnamine,
    SMILESTokenizerGuacaMol,
)
from acegen.vocabulary.vocabulary import SMILESVocabulary

tokenizer_options = {
    "AISTokenizer": AISTokenizer,
    "DeepSMILESTokenizer": DeepSMILESTokenizer,
    "SAFETokenizer": SAFETokenizer,
    "SELFIESTokenizer": SELFIESTokenizer,
    "SMILESTokenizerChEMBL": SMILESTokenizerChEMBL,
    "SMILESTokenizerEnamine": SMILESTokenizerEnamine,
    "SMILESTokenizerGuacaMol": SMILESTokenizerGuacaMol,
}
