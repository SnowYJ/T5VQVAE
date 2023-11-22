__author__ = 'Danilo S. Carvalho <danilo@jaist.ac.jp>, Vu Duc Tran <vu.tran@jaist.ac.jp>'

import regex as re

RGX_PUNCT = r" ([^\P{P}-])"

class PlainFormatter(object):
    @classmethod
    def dumps(cls, document):
        if(len(document.sentences) == 0):
            return "";

        output = []

        for sentence in document.sentences:
            if (len(document.sentences[0].terms) > 0):
                output.append(term.surface for term in sentence.terms)
            else:
                output.append(token.surface for token in sentence.tokens)

        return "\n".join((re.sub(RGX_PUNCT, r"\1", " ".join(sent)) for sent in output))


