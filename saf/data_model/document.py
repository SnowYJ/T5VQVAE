__author__ = 'danilo@jaist.ac.jp'

from typing import List, Tuple, Dict

from .annotable import Annotable
from .sentence import Sentence


class Document(Annotable):
    """A corpus document description.

    Attributes:
    meta (dict): Dictionary containing document metadata. Each key represents a meta attribute.
    title (unicode): The document title.
    sentences (list of Sentence): List of sentences in the document, in reading order.
    paragraphs (list of pair of int): [optional] List of pairs (start, end) of sentence indexes composing
        the document paragraphs, in reading order.
    annotations (dict): [optional] Document level annotations, identified by the dictionary key.
        Examples: document vector, topics (in a topic model).

    """
    def __init__(self):
        self.meta = dict()
        self.title: str = ""
        self.sentences: List[Sentence] = []
        self.paragraphs: List[Tuple[int, int]] = []
        self.annotations: Dict = dict()
