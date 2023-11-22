__author__ = "Danilo S. Carvalho <danilo@jaist.ac.jp>"

from abc import ABCMeta
from abc import abstractmethod


class Annotator(object, metaclass=ABCMeta):
    def __init__(self):
        """Formatter constructor

        Constructs an annotator object that includes or modify annotations for objects in the data model.

        :return: new Annotator instance.
        """
        pass


    @abstractmethod
    def annotate(self, annotable):
        """Annotates a document

        :param annotable (Annotable): object to be annotated.
        :return: Annotated Annotable (Document, Sentence, Term, Token, ...) object.
        """

        raise NotImplementedError()


class AnnotationError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
