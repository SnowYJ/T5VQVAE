import sys
from typing import Iterable
from zipfile import ZipFile
from saf import Sentence, Token
import pandas as pd
import random
from transformers import PreTrainedTokenizer
import torch
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
import os
import time
import pickle

DSR_FILES = {
    "wikipedia": "datasets/full/Wikipedia/WKP_DSR_model_CSV.zip",
    "wiktionary": "datasets/full/Wikitionary/WKT_DSR_model_CSV.zip",
    "wordnet": "datasets/full/WordNet/WN_DSR_model_CSV.zip"
}


class DefinitionSemanticRoleCorpus(Iterable[Sentence]):
    def __init__(self, path: str):
        if (path in DSR_FILES):
            path = DSR_FILES[path]

        dsr_zip = ZipFile(path)
        self._source = dsr_zip.open(dsr_zip.namelist()[0])

        self._size = 0

        for line in self._source:
            self._size += 1

        self._source.seek(0)

    def __iter__(self):
        return DefinitionSemanticRoleCorpusIterator(self)

    def __len__(self):
        return self._size


class DefinitionSemanticRoleCorpusIterator:
    def __init__(self, dsrc: DefinitionSemanticRoleCorpus):
        self._dsrc = dsrc
        self._sent_gen = self.sent_generator()

    def __next__(self):
        return next(self._sent_gen)

    def sent_generator(self):
        k = 0
        sentence_buffer = [None]
        while (sentence_buffer):
            sentence_buffer = list()
            while (not sentence_buffer):
                try:
                    line_bytes = next(self._dsrc._source)
                    line_bytes = line_bytes.replace(b"\r", b"\\r").replace(b"\t", b"\\t")
                    line_bytes = line_bytes.replace(b"\N", b"\\N").replace(b"\c", b"\\c")
                    line_bytes = line_bytes.replace(b"\i", b"\\i")
                    line = line_bytes.decode("unicode_escape")
                    line = line.strip().replace("&amp;", "&").replace("&quot;", "\"")
                    fields = line.split(";")
                    terms = fields[2].split(", ")

                    for term in terms:
                        sentence = Sentence()
                        sentence.annotations["id"] = fields[0]
                        sentence.annotations["POS"] = fields[1]
                        sentence.annotations["definiendum"] = term
                        sentence.annotations["definition"] = fields[3]

                        for i in range(4, len(fields)):
                            segment_role = fields[i].split("/")
                            segment = "/".join(segment_role[:-1])
                            role = segment_role[-1]
                            for tok in segment.split():
                                token = Token()
                                token.surface = tok
                                token.annotations["DSR"] = role
                                sentence.tokens.append(token)

                        sentence_buffer.append(sentence)

                except UnicodeDecodeError:
                    print("Decode error", file=sys.stderr)
                except StopIteration:
                    break

            for sentence in sentence_buffer:
                # print([t.surface for t in sentence.tokens])

                yield sentence


def load_sample(corpus: Iterable[Sentence], split: tuple = (.7, .2), randomize: bool = True):
    sents = list(corpus)
    if (randomize):
        random.shuffle(sents)

    train_cut = int(split[0] * len(sents))
    valid_cut = int((split[0] + split[1]) * len(sents))
    train, valid = sents[0:train_cut], sents[train_cut: valid_cut]
    test = sents[valid_cut:] if (split[0] + split[1] < 1) else []

    return train, valid, test


def load_dataset(dataset, encoder_tokenizer: PreTrainedTokenizer = None, decoder_tokenizer: PreTrainedTokenizer=None, set_seq_size=None):
    example = []
    for sent in tqdm(dataset):
        txt = [t.surface for t in sent.tokens]

        text_0 = ' '.join(txt)
        text_1 = '<BOS> ' + ' '.join(txt)
        text_2 = '<BOS> ' + ' '.join(txt) + ' <EOS>'

        cur_enc_len = len(encoder_tokenizer.encode(text_0))
        cur_dec_len = len(decoder_tokenizer.encode(text_0))

        if cur_enc_len >= set_seq_size or cur_dec_len >= set_seq_size:
            continue

        enc_input = encoder_tokenizer.batch_encode_plus([text_0], max_length=set_seq_size, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
        dec_input = decoder_tokenizer.batch_encode_plus([text_1], max_length=set_seq_size, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
        dec_label = decoder_tokenizer.batch_encode_plus([text_2], max_length=set_seq_size, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')

        example.append({'input_enc_ids': enc_input['input_ids'][0], 'input_dec_ids': dec_input['input_ids'][0], 'label_dec_ids': dec_label['input_ids'][0]})

    print("After filter, dataset size: ", len(example))

    return example


class SetSizeLineByLineTextDataset(Dataset):
    def __init__(
            self,
            encoder_tokenizer: PreTrainedTokenizer,
            decoder_tokenizer: PreTrainedTokenizer,
            dataset: str,
            set_seq_size: int
    ):
        self.examples = load_dataset(dataset, encoder_tokenizer, decoder_tokenizer, set_seq_size)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def get_dataset(tr_dataset, te_dataset, encoder_tokenizer, decoder_tokenizer, set_seq_size):
    return (SetSizeLineByLineTextDataset(dataset=tr_dataset, encoder_tokenizer=encoder_tokenizer, decoder_tokenizer=decoder_tokenizer, set_seq_size=set_seq_size),
            SetSizeLineByLineTextDataset(dataset=te_dataset, encoder_tokenizer=encoder_tokenizer, decoder_tokenizer=decoder_tokenizer, set_seq_size=set_seq_size))


if __name__ == '__main__':
    pass


