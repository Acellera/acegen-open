from acegen.vocabulary.base import BaseTokenizer, BaseVocabulary
from acegen.vocabulary.tokenizers import (
    AISTokenizer,
    AsciiSMILESTokenizer,
    DeepSMILESTokenizer,
    SAFETokenizer,
    SELFIESTokenizer,
    SMILESTokenizerChEMBL,
    SMILESTokenizerEnamine,
    SMILESTokenizerGuacaMol,
)
from acegen.vocabulary.vocabulary import Vocabulary

tokenizer_options = {
    "AISTokenizer": AISTokenizer,
    "DeepSMILESTokenizer": DeepSMILESTokenizer,
    "SAFETokenizer": SAFETokenizer,
    "SELFIESTokenizer": SELFIESTokenizer,
    "SMILESTokenizerChEMBL": SMILESTokenizerChEMBL,
    "SMILESTokenizerEnamine": SMILESTokenizerEnamine,
    "SMILESTokenizerGuacaMol": SMILESTokenizerGuacaMol,
    "AsciiSMILESTokenizer": AsciiSMILESTokenizer,
}
