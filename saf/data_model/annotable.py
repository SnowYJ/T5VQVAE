__author__ = "Danilo S. Carvalho <danilo@jaist.ac.jp>"

from typing import Dict


class Annotable(object):
    """A corpus annotable object.

    Attributes:
    annotations (dict): [optional] General type annotations, identified by the dictionary key.
        Examples: Linguistic, statistic, categorical, etc. annotations.

    """
    def __init__(self):
        self.annotations: Dict = dict()

    def typename(self):
        self.__class__.typename()
