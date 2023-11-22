__author__ = 'danilo@jaist.ac.jp'

from typing import List, Dict

from .annotable import Annotable
from .term import Term
from .token import Token


class Sentence(Annotable):
    """Description of a sentence.

    Attributes:
    tokens (list of Token): Sequence of tokens composing the sentence, in reading order.
    terms (list of Term): [optional] Sequence of terms (morphemes -> phrases) composing the sentence, in reading order.
    annotations (dict): [optional] Sentence level annotations, identified by the dictionary key.
        Examples: sentence vector, inclusion status (in extractive summarization).
    """

    def __init__(self):
        self.tokens: List[Token] = []
        self.terms: List[Term] = []
        self.annotations: Dict = dict()
