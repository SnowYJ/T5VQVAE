# -*- coding: utf-8 -*-
__author__ = 'Danilo S. Carvalho <danilo@jaist.ac.jp>, Vu Duc Tran <vu.tran@jaist.ac.jp>'

from saf.data_model.document import Document
from saf.data_model.sentence import Sentence
from saf.data_model.token import Token
from saf.constants import annotation
from saf.importers.conll import CoNLLImporter
from saf.formatters.conll import CoNLLFormatter
from saf.formatters.plain import PlainFormatter


from saf.importers.plain import PlainImporter
from saf.importers.conll import CoNLLImporter
from saf.constants import annotation
from nltk.tokenize import sent_tokenize, word_tokenize

import unittest

plain_doc = """
But I must explain to you how all this mistaken idea of denouncing of a pleasure and praising pain was born and I will give you a complete account of the system, and expound the actual teachings of the great explorer of the truth, the master-builder of human happiness.
 No one rejects, dislikes, or avoids pleasure itself, because it is pleasure, but because those who do not know how to pursue pleasure rationally encounter consequences that are extremely painful.
 Nor again is there anyone who loves or pursues or desires to obtain pain of itself, because it is pain, but occasionally circumstances occur in which toil and pain can procure him some greatpleasure.
 To take a trivial example, which of us ever undertakes laborious physical exercise, except to obtain some advantage from it? But who has any right to find fault with a man who chooses to enjoy a pleasure that has no annoying consequences, or one who avoids a pain that produces no resultant pleasure?
On the other hand, we denounce with righteous indignation and dislike men who are so beguiled and demoralized by the charms of pleasure of the moment, so blinded by desire, that they cannot foresee the pain and trouble that are bound to ensue; and equal blame belongs to those who fail in their duty through weakness of will, which is the same as saying through shrinking from toil and pain.
 These cases are perfectly simple and easy to distinguish.
 In a free hour, when our power of choice is untrammeled and when nothing prevents our being able to do what we like best, every pleasure is to be welcomed and every pain avoided.
 But in certain circumstances and owing to the claims of duty or the obligations of business it will frequently occur that pleasures have to be repudiated and annoyances accepted.
 The wise man therefore always holds in these matters to this principle of selection: he rejects pleasures to secure other greater pleasures, or else he endures pains to avoid worse pains.
"""

conll_doc = """
# sent_id = dev-s3
# text = A Nina é a chance dele ser feliz.
1   A   _   DET DET _   2   det _   _
2   Nina    _   PROPN   PNOUN   _   5   nsubj   _   _
3   é  _   VERB    VERB    _   5   cop _   _
4   a   _   DET DET _   5   det _   _
5   chance  _   NOUN    NOUN    _   0   root    _   _
6-7 dele    _   _   _   _   _   _   _   _
6   de  _   ADP ADP _   8   mark    _   _
7   ele _   PRON    PRON    _   8   nsubj   _   _
8   ser _   VERB    VERB    _   5   nmod    _   _
9   feliz   _   ADJ ADJ _   8   xcomp:adj   _   SpaceAfter=No
10  .   _   PUNCT   .   _   5   punct   _   _

# sent_id = dev-s4
# text = É um dos três ácidos ftálicos isoméricos.
1   É   _   VERB    VERB    _   2   cop _   _
2   um  _   PRON    PRON    _   0   root    _   _
3-4 dos _   _   _   _   _   _   _   _
3   de  de  ADP ADP _   6   case    _   _
4   os  o   DET DET Definite=Def|Gender=Masc|Number=Plur|PronType=Art   6   det _   _
5   três   _   NUM NUM NumType=Card    6   nummod  _   _
6   ácidos _   NOUN    NOUN    _   2   nmod    _   _
7   ftálicos   _   ADJ ADJ _   6   amod    _   _
8   isoméricos _   ADJ ADJ _   6   amod    _   SpaceAfter=No
9   .   _   PUNCT   .   _   2   punct   _   _
"""

class TestCoNLLFormatter(unittest.TestCase):
    def test_conll_format(self):

        conll_importer = CoNLLImporter(field_list=[annotation.LEMMA, annotation.UPOS, annotation.POS])
        doc = conll_importer.import_document(conll_doc)
        self.assertEqual(len(doc.sentences),2)
        self.assertEqual(len(doc.sentences[0].terms),9)
        self.assertEqual(len(doc.sentences[0].tokens),10)

        self.assertEqual(len(doc.sentences[1].terms),8)
        self.assertEqual(len(doc.sentences[1].tokens),9)

        conll_formatter = CoNLLFormatter(field_list=[annotation.LEMMA, annotation.UPOS, annotation.POS])
        conll_formatted_doc = conll_formatter.dumps_wo_term_w_id(doc)

        print(conll_formatted_doc)

    def test_conll_format_w_plain_input(self):
        plain_importer = PlainImporter(sent_tokenize, word_tokenize)
        doc = plain_importer.import_document(plain_doc)

        self.assertEqual(len(doc.sentences), 10)
        self.assertEqual(doc.sentences[0].tokens[-1].surface, ".")

        conll_formatter = CoNLLFormatter(field_list=[annotation.LEMMA, annotation.UPOS, annotation.POS])
        conll_formatted_doc = conll_formatter.dumps(doc)

        print(conll_formatted_doc)

        print(PlainFormatter.dumps(doc))
