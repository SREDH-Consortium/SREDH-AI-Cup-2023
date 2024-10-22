from typing import Iterable, Dict, Union

import spacy
from spacy.util import compile_prefix_regex, compile_suffix_regex, compile_infix_regex
import string
from spacy.tokenizer import Tokenizer



class SpacySentencizer(object):
    """
    This class is used to read text and split it into 
    sentences (and their start and end positions)
    using a spacy model
    """

    def __init__(self, spacy_model: str):
        """
        Initialize a spacy model to read text and split it into 
        sentences.
        Args:
            spacy_model (str): Name of the spacy model
        """
        # fixes = [r'['+string.punctuation+']',r'(?<=[a-z])(?=[n]$)',r'(?<=\d)(?=\d{2}/)',r'(?<=[A-Z])(?=[a-z])',r'(?<=0)(?=\d)', r'(?<=\d)(?=0)',r'(?<=[a-z])(?=[A-Z])',r'(?<=[a-zA-Z])(?=\d)',r'(?<=\d)(?=[a-zA-Z])']  # 匹配逗号和分号作为infix
        # special_fix = [r'(?<=[A-Za-z])(?=at)',r'(?<=Dr)(?=[A-Za-z])',r'(?<=DR)(?=[A-Za-z])',r'(?<=[A-Za-z])(?=report)',r'(?<=day)',r'(?=[\d]+)',r'(?=[A-Z])+']
        # # ,r'(?=[\d]+)',r'(?=[A-Z])+'
        # self.fixes = fixes+special_fix
        self._nlp = spacy.load(spacy_model)

    def get_sentences(self, text: str) -> Iterable[Dict[str, Union[str, int]]]:
        """
        Return an iterator that iterates through the sentences in the text
        Args:
            text (str): The text
        Returns:
            (Iterable[Dict[str, Union[str, int]]]): Yields a dictionary that contains the text of the sentence
                                                    the start position of the sentence in the entire text
                                                    and the end position of the sentence in the entire text
        """
        # self._nlp.tokenizer = Tokenizer(
        #     self._nlp.vocab,
        #     prefix_search=compile_prefix_regex(self.fixes).search,
        #     suffix_search=compile_suffix_regex(self.fixes).search,
        #     infix_finditer=compile_infix_regex(self.fixes).finditer
        # )
        document = self._nlp(text)
        for sentence in document.sents:
            yield {'text': sentence.text,
                   'start': sentence.start_char,
                   'end': sentence.end_char,
                   'last_token': None}
