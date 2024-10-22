import spacy
from typing import Tuple, Iterable, Mapping, Dict, Union

from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_suffix_regex, compile_infix_regex
import string


class SpacyTokenizer(object):
    """
    This class is used to read text and return the tokens
    present in the text (and their start and end positions)
    using spacy
    """

    def __init__(self, spacy_model: str):
        """
        Initialize a spacy model to read text and split it into 
        tokens.
        Args:
            spacy_model (str): Name of the spacy model
        """
        
        # F1:87.4 chunck_size:32
        # fixes = [r'['+string.punctuation+']',r'[0]',r'(?<=[a-z])(?=\d)',r'(?<=\d)(?=[a-z])']
        # special_fix = [r'(?<=[Aa][Tt])',r'(?<=[Tt][Oo])',r'(?<=[a-z])(?=[n]$)',r'(?<=[Dd][Rr])',r'(?<=[Hh][Oo][Ss][Pp][Ii][Tt][Aa][Ll])',r'(?<=[Rr][Ee][Pp][Oo][Rr][Tt])',r'(?<=[Dd][Aa][Yy])']
        
        
        # special_fix = [r'(?=[\d]+)',r'(?=[A-Z])+',r'[Rr][Ee][Pp][Oo][Rr][Tt]',r'[Dd][Aa][Yy]'] #
        # special_fix = [r'(?=[\d]+)',r'(?=[A-Z])+',r'(?=[a-z])+',r'[Rr][Ee][Pp][Oo][Rr][Tt]',r'[Dd][Aa][Yy]']
        # fixes = [r'['+string.punctuation+']',r'(?<=[a-z])(?=[n]$)',r'(?<=\d)(?=\d{2}/)',r'(?<=[A-Z])(?=[a-z])',r'(?<=0)(?=\d)', r'(?<=\d)(?=0)',r'(?<=[a-z])(?=[A-Z])',r'(?<=[a-zA-Z])(?=\d)',r'(?<=\d)(?=[a-zA-Z])']
        # # ,r'(?=[\d]+)',r'(?=[A-Z])+',r'(?=[a-z])+'

        # Formal
        fixes = [r'['+string.punctuation+']',r'[0]',r'(?<=[A-Za-z])(?=\d)',r'(?<=\d)(?=[A-Za-z])',r'\n',r'(?<=\d)(?=\d{2}/)',r'(?<=[A-Z])(?=[a-z])',r'(?<=[a-z])(?=[A-Z])']
        special_fix = [r'[Aa][Tt]',r'[Tt][Oo]',r'[Dd][Rr]',r'[Hh][Oo][Ss][Pp][Ii][Tt][Aa][Ll]',r'[Rr][Ee][Pp][Oo][Rr][Tt]',r'[Dd][Aa][Yy]']
        #lower_continue
        # fixes = [r'['+string.punctuation+']',r'[0]',r'(?<=[A-Za-z])(?=\d)',r'(?<=\d)(?=[A-Za-z])',r'\n',r'(?<=\d)(?=\d{2}/)',r'(?<=[A-Z])(?=[a-z])',r'(?<=[a-z])(?=[A-Z])']
        # special_fix = [r'(?=[\d]+)',r'(?=[A-Z])+',r'[Rr][Ee][Pp][Oo][Rr][Tt]',r'[Dd][Aa][Yy]']
        #all_split
        # fixes = [r'['+string.punctuation+']',r'[0]',r'(?<=[A-Za-z])(?=\d)',r'(?<=\d)(?=[A-Za-z])',r'\n',r'(?<=\d)(?=\d{2}/)',r'(?<=[A-Z])(?=[a-z])',r'(?<=[a-z])(?=[A-Z])']
        # special_fix = [r'(?=[\d]+)',r'(?=[A-Z])+',r'(?=[a-z])+',r'[Rr][Ee][Pp][Oo][Rr][Tt]',r'[Dd][Aa][Yy]']

        self.fixes = fixes+special_fix
        # self.fixes = fixes
        self._nlp = spacy.load(spacy_model)

        # self._nlp = spacy.load(spacy_model)

    @staticmethod
    def __get_start_and_end_offset(token: spacy.tokens.Token) -> Tuple[int, int]:
        """
        Return the start position of the token in the entire text
        and the end position of the token in the entire text
        Args:
            token (spacy.tokens.Token): The spacy token object
        Returns:
            start (int): the start position of the token in the entire text
            end (int): the end position of the token in the entire text
        """
        start = token.idx
        end = start + len(token)
        return start, end

    def get_tokens(self, text: str) -> Iterable[Dict[str, Union[str, int]]]:
        """
        Return an iterable that iterates through the tokens in the text
        Args:
            text (str): The text to annotate
        Returns:
            (Iterable[Mapping[str, Union[str, int]]]): Yields a dictionary that contains the text of the token
                                                       the start position of the token in the entire text
                                                       and the end position of the token in the entire text
        """
        self._nlp.tokenizer = Tokenizer(
            self._nlp.vocab,
            prefix_search=compile_prefix_regex(self.fixes).search,
            suffix_search=compile_suffix_regex(self.fixes).search,
            infix_finditer=compile_infix_regex(self.fixes).finditer
        )
        document = self._nlp(text)
        for token in document:
            start, end = SpacyTokenizer.__get_start_and_end_offset(token)
            yield {'text': token.text, 'start': start, 'end': end}
