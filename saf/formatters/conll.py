__author__ = 'Danilo S. Carvalho <danilo@jaist.ac.jp>, Vu Duc Tran <vu.tran@jaist.ac.jp>'

from saf.constants import annotation

class CoNLLFormatter(object):
    def __init__(self, field_list):
        self.field_list = field_list

    # def dumps(self, document):
    #     output = []
    #     for sentence in document.sentences:
    #         for token in sentence.tokens:
    #             output.append([token.surface] + [token.annotations[field] if (field in token.annotations) else u"-" for field in self.field_list])
    #         output.append([])
    #
    #     return u"\n".join((u"\t".join(line) for line in output))

    # def dump(self, document, file):
    #     for sentence in document.sentences:
    #         for token in sentence.tokens:
    #             file.write(u"\t".join([token.surface] + [token.annotations[field] if (field in token.annotations) else u"-" for field in self.field_list]))
    #             file.write(u"\n")
    #
    #         file.write(u"\n")


    def dumps(self, document):
        if(len(document.sentences) == 0):
            return ""

        has_term = len(document.sentences[0].terms) > 0

        has_id = annotation.ID in document.sentences[0].tokens[0].annotations

        header = "# SURFACE\t" + "\t".join(self.field_list) + "\n"

        if(has_term and has_id):
            return header + self.dumps_w_term_w_id(document)
        elif(has_term and not has_id):
            return header + self.dumps_w_term_wo_id(document)
        elif(not has_term and has_id):
            return header + self.dumps_wo_term_w_id(document)
        elif(not has_term and not has_id):
            return header + self.dumps_wo_term_wo_id(document)
        else:
            pass


    def dumps_w_term_w_id(self, document):
        output = []
        for sentence in document.sentences:
            for term in sentence.terms:
                if(len(term.tokens) > 1):
                    output.append(
                        [term.annotations[annotation.ID], term.surface]
                        + [term.annotations[field] if (field in term.annotations) else "_" for field in self.field_list])

                for token in term.tokens:
                    output.append([token.annotations[annotation.ID], token.surface]
                                  + [token.annotations[field] if (field in token.annotations) else "_" for field in self.field_list])

            output.append([])

        return "\n".join(("\t".join(line) for line in output))

    def dumps_w_term_wo_id(self, document):
        output = []

        for sentence in document.sentences:
            last_token_id = 0
            for term in sentence.terms:
                if(len(term.tokens) > 1):
                    output.append(
                        ["%d-%d" % (last_token_id+1, last_token_id+len(term.tokens)), term.surface]
                        + [term.annotations[field] if (field in term.annotations) else "_" for field in self.field_list])

                for token in term.tokens:

                        output.append([str(last_token_id + 1), token.surface]
                                      + [token.annotations[field] if (field in token.annotations) else "_" for field in self.field_list])
                        last_token_id += 1

            output.append([])

        return "\n".join(("\t".join(line) for line in output))

    def dumps_wo_term_w_id(self, document):
        output = []
        for sentence in document.sentences:
            for token in sentence.tokens:
                output.append([token.annotations[annotation.ID], token.surface] + [token.annotations[field] if (field in token.annotations) else "_" for field in self.field_list])

            output.append([])

        return "\n".join(("\t".join(line) for line in output))

    def dumps_wo_term_wo_id(self, document):
        output = []

        for sentence in document.sentences:
            last_token_id = 0
            for token in sentence.tokens:
                output.append([str(last_token_id+1),token.surface] + [token.annotations[field] if (field in token.annotations) else "_" for field in self.field_list])
                last_token_id += 1
            output.append([])

        return "\n".join(("\t".join(line) for line in output))


