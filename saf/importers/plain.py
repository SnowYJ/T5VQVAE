__author__ = 'Danilo S. Carvalho <danilo@jaist.ac.jp>, Vu Duc Tran <vu.tran@jaist.ac.jp>'

from saf.data_model.document import Document
from saf.data_model.sentence import Sentence
from saf.data_model.token import Token
from .importer import Importer


class PlainImporter(Importer):
    def import_document(self, document):
        doc = Document()

        sentences_raw = self.sent_tokenizer(document)

        for sent_raw in sentences_raw:
            tokens_raw = self.word_tokenizer(sent_raw)

            sentence = Sentence()

            for token_raw in tokens_raw:
                token = Token()
                token.surface = token_raw
                sentence.tokens.append(token)

            doc.sentences.append(sentence)

        return doc
