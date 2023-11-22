__author__ = 'Danilo S. Carvalho <danilo@jaist.ac.jp>, Vu Duc Tran <vu.tran@jaist.ac.jp>'

import re

from saf.data_model.document import Document
from saf.data_model.sentence import Sentence
from saf.data_model.token import Token
from .importer import Importer
from saf.importers.tokenizers.conll import conll_sentence_tokenize, conll_word_tokenize


class WebAnnoImporter(Importer):
    def __init__(self, field_list):
        """Importer constructor

        Constructs an Importer object that converts a WebAnno TSV3 document into an Annotable document.

        :param field_list (list of str): list of keys identifying the fields in the WebAnno TSV3 file.
            Default keys can be found in the constants.annotation module.
            Example: ["POS", "DEP", ...]
        :return: new Formatter instance.
        """

        self.sent_tokenizer = conll_sentence_tokenize
        self.word_tokenizer = conll_word_tokenize
        self.field_list = field_list

    def import_document(self, document):

        doc = Document()

        sentences_raw = self.sent_tokenizer(document)

        for sent_raw in sentences_raw:
            tokens_raw = self.word_tokenizer(sent_raw)

            sentence = Sentence()
            last_term_id = (-1,)

            for token_raw in tokens_raw:
                if(token_raw.startswith("#")):
                    continue

                token = Token()

                fields = token_raw.split()
                sent_tok_idx = fields[0]
                char_span = fields[1]
                token.surface = fields[2]
                annotations = dict(zip(self.field_list, fields[3:]))

                for field in self.field_list:
                    annotations[field] = annotations[field].split("|")
                    for i in range(len(annotations[field])):
                        mo = re.match(r"(?P<name>.+)\[\d+\]$", annotations[field][i])

                        if (mo):
                            annotations[field][i] = mo.group("name").replace("\\", "")
                        else:
                            annotations[field][i] = annotations[field][i].replace("\\", "")

                token.annotations = annotations
                sentence.tokens.append(token)

            if(len(sentence.tokens) == 0):
                continue

            doc.sentences.append(sentence)

        return doc
