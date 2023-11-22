import numpy as np
import pandas as pd


def frange_cycle_zero_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio_increase=0.5, ratio_zero=0.3):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio_increase) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            if i < period*ratio_zero:
                L[int(i+c*period)] = start
            else:
                L[int(i+c*period)] = v
                v += step
            i += 1
    return L


def create_srl_vocab(dataset):
    vocab = dict()
    dat = pd.read_csv(dataset, sep="&", header=None, names=["text", "role"])
    seq_texts = dat.values.tolist()

    for text in seq_texts:
        label = text[1].strip().split(" ")
        for i in label:
            if i not in vocab:
                vocab[i] = 1
            else:
                vocab[i] += 1

    # sort according to num.
    sorted_dict = sorted(vocab.items(), key=lambda item: item[1], reverse=True)
    order_vocab = dict()
    for i, v in enumerate(sorted_dict):
        order_vocab[v[0]] = i

    order_vocab['FIX'] = i+1
    order_vocab['PAD'] = i+2

    return order_vocab


if __name__ == '__main__':
    pass