from train_optimus import load_optimus
import argparse
import os
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu
import pickle
from optimus.utils import set_seed, logging, load_sample, conv_sent_dict, strip_eos, convert, millify, tqdm_joblib, load_sample_inf, load_sample_def
from optimus.batchify import get_batches, get_batches_lm
from optimus.pytorch_transformers import (BertConfig, BertForLatentConnector, BertTokenizer, GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer)
import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats
import math
import logging
import load_definition as definition
import random
from nltk.corpus import stopwords
from nltk import download
import gensim.downloader as api
from optimus.examples.big_ae.run_latent_generation import interpolate


MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer),
    'bert': (BertConfig, BertForLatentConnector, BertTokenizer)
}


def generation_optimus(model, tokenizer_decoder, inputs, args=None):
    attention_mask=(inputs > 0).float()
    outputs = model.encoder(inputs.long(), attention_mask, role_ids=None)
    pooled_hidden_fea = outputs[1]  # model outputs are always tuple in pytorch-transformers (see doc)
    latent_z, _ = model.connect(pooled_hidden_fea)
    past = latent_z.squeeze(1)

    context_tokens = tokenizer_decoder.encode('<BOS>')
    args.top_k = 0
    args.top_p = 1.0
    args.temperature = 1
    length = 40

    out = sample_sequence_conditional(
        model=model.decoder,
        context=context_tokens,
        past=past,
        length=length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=args.device,
        decoder_tokenizer=tokenizer_decoder
    )

    text_x1 = tokenizer_decoder.decode(out[0, :].tolist()) # , clean_up_tokenization_spaces=True
    text_x1 = text_x1.split()
    text_x1 = ' '.join(text_x1[1:])

    return text_x1


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


# def sample_sequence_conditional(model, length, context, past=None, num_samples=1, temperature=1, top_k=0, top_p=0.0, device='cpu', decoder_tokenizer=None, cvae=False, args=None):
#     context = torch.tensor(context, dtype=torch.long, device=device)
#     context = context.unsqueeze(0).repeat(num_samples, 1)
#     generated = context
#     i=0
#     with torch.no_grad():
#         while i<length:
#             inputs = {'input_ids': generated, 'encoder_hidden_states': past}
#             if args.model_type == 't5':
#                 sequence_output = model.decoder(**inputs)[0]  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
#                 sequence_output = sequence_output * (model.model_dim ** -0.5)
#             elif args.model_type == 'bart':
#                 sequence_output = model.model.decoder(**inputs)[0]  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
#                 sequence_output = sequence_output * (768 ** -0.5)
#             else:
#                 exit()
#
#             outputs = model.lm_head(sequence_output)
#             next_token_logits = outputs[0, -1, :] / temperature
#             filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
#             next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
#             generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
#
#             # print(next_token.unsqueeze(0)[0,0].item())
#             if next_token.unsqueeze(0)[0,0].item() == :
#                 break
#
#             i+=1
#
#     return generated

def sample_sequence_conditional(model, length, context, past=None, num_samples=1, temperature=1, top_k=0, top_p=0.0, device='cpu', decoder_tokenizer=None, cvae=False):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    i=0
    with torch.no_grad():
        while i<length:
            # for _ in trange(length):
            inputs = {'input_ids': generated, 'past': past}
            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            next_token_logits = outputs[0][0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

            if next_token.unsqueeze(0)[0,0].item() == decoder_tokenizer.encode('<EOS>')[0]:
                break

            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)

            i+=1

    return generated


# --------------------------------------------------- Traversal --------------------------------------------------------
class Traverser:
    def __init__(self, dim):
        self.dim = dim

    def traverse_continuous_line(self, idx, size, loc=0, scale=1):
        samples = np.zeros(shape=(size, self.dim))
        if idx is not None:
            cdf_traversal = np.linspace(0.2, 0.8, size)
            cont_traversal = stats.norm.ppf(cdf_traversal, loc=loc, scale=scale)
            # cont_traversal = stats.norm.ppf(cdf_traversal)
            for i in range(size):
                samples[i, idx] = cont_traversal[i]
        return samples.astype('f')

    def traverse_continuous_line_control(self, idx, size, loc=0, scale=1, v=0, direct='left'):
        samples = np.zeros(shape=(size, self.dim))
        if idx is not None:
            prob = stats.norm.cdf(v, loc=loc, scale=scale)
            if direct == 'left':
                cdf_traversal = np.linspace(0.000000001, prob, size)
            else:
                cdf_traversal = np.linspace(prob, 0.999999999, size)

            cont_traversal = stats.norm.ppf(cdf_traversal, loc=loc, scale=scale)

            for i in range(size):
                samples[i, idx] = cont_traversal[i]
        return samples.astype('f'), sum(cont_traversal)/len(cont_traversal)


def decode_optimus(z, latent_z=None, model=None, tokenizer_decoder=None):
    """
    z: latent output of optimus
    latent_z: latent output of optimus given input.
    """

    # input = torch.tensor([tokenizer_decoder.bos_token_id for _ in range(args.num_sent)]).view(args.num_sent, -1)

    if latent_z is not None:
        # find the conlumn index with nonzero value and then replace by
        input_latent = torch.cat([latent_z for _ in range(args.num_sent)], 0)
        column_index = np.nonzero(np.array(z))[1][0]
        input_latent[:, column_index] = torch.tensor(z)[:, column_index]
    else:
        input_latent = torch.tensor(z).view(z.shape[0], -1)

    inputs, sents = [], []

    from optimus.examples.big_ae.run_latent_generation import text_from_latent_code

    args.top_k = 0
    args.top_p = 1.0
    args.temperature = 1.0

    for i in input_latent:
        res = text_from_latent_code(i.view(1, -1), model, args, tokenizer_decoder)
        sents.append(res)

    return sents


def traverse(seed, tokenizer_encoder, tokenizer_decoder, model):
    dim_z = 32
    args.num_sent = 4
    seed = tokenizer_encoder.convert_tokens_to_ids(seed.split())
    encode_input = torch.tensor(seed).unsqueeze(0)
    attention_mask = (encode_input > 0).float()
    outputs = model.encoder(encode_input.long(), attention_mask)[1]
    latent_z, _, mean, logvar = model.connect_traversal(outputs)
    latent_z = latent_z.squeeze(1)
    print("Origin: ", decode_optimus(latent_z, model=model, tokenizer_decoder=tokenizer_decoder))

    for i in np.arange(dim_z, step=1):
        # randomly choose four value from normal distribution where the mean and variance from model.
        loc, scale = mean[i], math.sqrt(math.exp(logvar[i]))
        # loc, scale = 0, 1
        samples = Traverser(dim_z).traverse_continuous_line(idx=i, size=args.num_sent, loc=loc, scale=scale)
        res = decode_optimus(samples, latent_z=latent_z, model=model, tokenizer_decoder=tokenizer_decoder)

        for ix, r in enumerate(res):
            print('Dim {}, Sent {}: {}'.format(i, ix, r))


# ---------------------------------------------------- Interpolation ---------------------------------------------------

def preprocess(sentence, stop_words):
    return [w for w in sentence.lower().split() if w not in stop_words]


def word_mover_distance(sent1, sent2, model, stopword):
    sent1 = preprocess(sent1, stopword)
    sent2 = preprocess(sent2, stopword)
    distance = model.wmdistance(sent1, sent2)
    return distance


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(args.save_dir, "train_log.log"))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)

    tokenizer_encoder, tokenizer_decoder, model = load_optimus(args, logger)
    model = model.to(device)

    # ----------------------------------------------- Traversal --------------------------------------------------------

    # traverse('an automobile is a kind of vehicle', tokenizer_encoder, tokenizer_decoder, model)
    # exit()

    # ----------------------------------------------- Arithmetic -------------------------------------------------------

    # args.sent_source = "the milky way is a kind of galaxy"
    # args.sent_target = "a rock is usually a solid"
    # args.top_k = 0
    # args.top_p = 1.0
    # args.temperature = 1
    # analogy(model, tokenizer_encoder, tokenizer_decoder, args)
    # exit()
    # ----------------------------------------------- Interpolation ----------------------------------------------------
    # args.top_k = 0
    # args.top_p = 1.0
    # args.temperature = 1
    # args.num_interpolation_steps = 10
    # args.sent_source = "some birds have a speckled brown color"
    # args.sent_target = "the weather balloon expands because the air pressure decreases at higher altitude"
    #
    # interpolate(model, tokenizer_encoder, tokenizer_decoder, args, control=None)
    # exit()
    # ----------------------------------------------- Interpolation Smoothness -----------------------------------------

    # model_wmd = api.load('word2vec-google-news-300')
    # download('stopwords')  # Download stopwords list.
    # stop_words = stopwords.words('english')
    # open_file = open('explanations.pkl', "rb")
    # data = pickle.load(open_file)
    # open_file.close()
    # random.shuffle(data)
    #
    # args.top_k = 0
    # args.top_p = 1.0
    # args.temperature = 1
    # args.num_interpolation_steps = 10
    #
    # IS_all = []
    # for i in range(0, len(data), 2):
    #     list_d = []
    #     print('----- Num: {} -----'.format(i))
    #     source, target = data[i], data[i+1]
    #     d_origin = word_mover_distance(source, target, model_wmd, stop_words)
    #     print('* source: ', source)
    #     print('* target: ', target)
    #     print('* distance: ', d_origin)
    #
    #     args.sent_source = source
    #     args.sent_target = target
    #     interpolate_path = interpolate(model, tokenizer_encoder, tokenizer_decoder, args, control=None)
    #     for j in range(len(interpolate_path)-1):
    #         d = word_mover_distance(interpolate_path[j], interpolate_path[j+1], model_wmd, stop_words)
    #         list_d.append(d)
    #
    #     IS = d_origin / sum(list_d)
    #     print('* IS: ', IS)
    #     IS_all.append(IS)
    #
    # print('* AVG IS: ', sum(IS_all)/len(IS_all))
    # print('* MAX IS: ', max(IS_all))
    # print('* MIN IS: ', min(IS_all))
    # exit()

    # ------------------------------------------------------------------------------------------------------------------

    # bleurt score
    bleurt_tokenizer = AutoTokenizer.from_pretrained("Elron/bleurt-base-512")
    bleurt_model = AutoModelForSequenceClassification.from_pretrained("Elron/bleurt-base-512")
    bleurt_model.eval()

    # sentenceT5 similarity
    sentenceT5_model = SentenceTransformer(model_name_or_path="checkpoints/checkpoint-sentence-transformers_sentence-t5-base") # 'sentence-transformers/sentence-t5-base'

    corpus = definition.DefinitionSemanticRoleCorpus(args.corpus)
    infer_type = False if args.inference_premises_sep else True
    train, valid, test = load_sample_def(corpus)

    train = train[:100]
    valid = valid[:100]

    conv_sent_dict_param = {'emb_tokenizer': tokenizer_encoder, 'decode_tokenizer': None if args.model == 'rnn' else tokenizer_decoder, 'model': args.model, 'args': args}

    train_sents, valid_sents = [], []
    for sent in tqdm(train):
        bert_list = tokenizer_encoder.tokenize(' '.join([t.surface for t in sent.tokens]))
        gpt_list = tokenizer_decoder.tokenize(' '.join([t.surface for t in sent.tokens]))

        if len(bert_list) > 25 or len(gpt_list) > 25:
            continue

        tr_temp = conv_sent_dict(sent, **conv_sent_dict_param)
        train_sents.append(tr_temp)

    for sent in tqdm(valid):

        bert_list = tokenizer_encoder.tokenize(' '.join([t.surface for t in sent.tokens]))
        gpt_list = tokenizer_decoder.tokenize(' '.join([t.surface for t in sent.tokens]))

        if len(bert_list) > 25 or len(gpt_list) > 25:
            continue

        val_temp = conv_sent_dict(sent, **conv_sent_dict_param)
        valid_sents.append(val_temp)

    train_par = {'data': train_sents, 'vocab': None, 'batch_size': args.batch_size, 'device': device, 'type_': args.exp, 'model': args.model, 'infer_type': infer_type,
                 'tokenizer_encoder': tokenizer_encoder if args.model in ['conditional_optimus'] else None,
                 'tokenizer_decoder': tokenizer_decoder if args.model in ['conditional_optimus'] else None}
    train_batches, _ = get_batches(**train_par)

    val_par = {'data': valid_sents, 'vocab': None, 'batch_size': args.batch_size, 'device': device, 'type_': args.exp, 'model': args.model, 'infer_type': infer_type,
               'tokenizer_encoder': tokenizer_encoder if args.model in ['conditional_optimus'] else None,
               'tokenizer_decoder': tokenizer_decoder if args.model in ['conditional_optimus'] else None}
    valid_batches, _ = get_batches(**val_par)

    model.eval()
    batches = valid_batches
    index = 0
    scores_sum_cos, scores_sum_bleurt, scores_sum_bleu = 0, 0, 0
    with torch.no_grad():
        for i in range(len(batches)):
            inputs, labels = batches[i][0].T, batches[i][1].T
            input_roles, label_roles = ((batches[i][2].T).long(), (batches[i][3].T).long()) if args.exp not in ['exp1', 'exp4_train', 'exp4_gen', 'exp_infer'] else (None, None)
            inputs_1 = batches[i][2].T if args.inference_premises_sep else None

            print('#########')
            gold_con = tokenizer_decoder.decode(labels.tolist()[0][1:-2], clean_up_tokenization_spaces=True)
            pred_con = generation_optimus(model, tokenizer_decoder, inputs, args=args)

            print(gold_con)
            print(pred_con)

            references = [gold_con]
            candidates = [pred_con]
            with torch.no_grad():
                bleurt_scores = bleurt_model(**bleurt_tokenizer(references, candidates, return_tensors='pt'))[0].squeeze().item()

            # SentenceT5 cosine score
            sentences = [pred_con, gold_con]
            embeddings = sentenceT5_model.encode(sentences)
            embed1 = torch.FloatTensor(embeddings[0])
            embed2 = torch.FloatTensor(embeddings[1])
            cos_scores = torch.cosine_similarity(embed1, embed2, dim=0)

            # BLEU score
            references = [gold_con.split(' ')]
            candidates = pred_con.split(' ')
            bleu_scores = sentence_bleu(references, candidates, weights=(1, 0, 0, 0))

            index += 1
            scores_sum_cos += cos_scores
            scores_sum_bleu += bleu_scores
            scores_sum_bleurt += bleurt_scores

            # loss_rec, loss_kl, loss = model.autoenc(inputs.long(), labels.long(), input_roles, label_roles, args, None, 0, infer=True if args.exp == 'exp_infer' else False, sep_infer_input=inputs_1)
            #
            # if (i + 1) % args.log_interval == 0:
            #     print('| epoch {:3d} | {:5d}/{:5d} batches | test loss {:.4f} , kl {:.4f} , latent rec {:.4f}'.format(1, i, len(batches), loss, loss_kl, loss_rec))
            #
            # final_loss += loss
    print("bleu: ", scores_sum_bleu/index)
    print("bleurt: ", scores_sum_bleurt/index)
    print("cosine: ", scores_sum_cos/index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    dic = {
        'batch_size':1,
        'cont_capacity':'0.0,5.0,25000.0,30.0',
        'corpus':'wordnet',
        'dec':'greedy',
        'decay_factor':0.1,
        'decay_patience':0,
        'dim_d':512,
        'dim_emb':512,
        'dim_h':1024,
        'dim_z':256,
        'latent_size': 768,
        'model_loss_func': 'beta_vae',
        'gradient_accumulation_steps': 1,
        'learning_rate': 5e-05,
        'device': 'cuda',
        'model': 'optimus',
        'latent_as_gpt_memory': True,
        'latent_as_gpt_emb': False,
        'use_pretrained_optimus': True,
        'pretrain_model_path': './checkpoint_optimus_mem_beta_0.0_latent_768_epoch_10_wordnet_dm',
        'inference_premises_com': True,
        'inference_premises_sep': False,
        'dim_target_kl': 1.0,
        'fb_mode': 0,
        'length_weighted_loss': False,
        'beta': 1.0,
        'disc_capacity':'0.0,5.0,25000.0,30.0',
        'dropout': 0.5,
        'epochs': 20,
        'eval_dis': False,
        'eval_interval':1,
        'exp':'exp4_gen', # LM: exp1, DM: exp4_gen
        'input_eval':None,
        'input_train':None,
        'lambda_adv':0,
        'lambda_kl':0,
        'lambda_p':0,
        'latent_spec':{'cont': 10, 'disc': [20, 2, 2, 3]},
        'lm':None,
        'lm_ckpt':None,
        'load_model':'',
        'local_rank':-1,
        'log_dir':None,
        'log_interval':20,
        'lr':0.0005,
        'max_len':20,
        'model_type':'beta',
        'nlayers':1,
        'no_cuda':False,
        'noise':[0.0, 0.0, 0.0, 0.0],
        'pretrain':False,
        'print_loss':False,
        'print_traversal':False,
        'pt_lm':'t5-small',
        'save_dir': './checkpoint_optimus_mem_beta_0.0_latent_768_epoch_10_wordnet_dm',
        'seed':0,
        'w2v_weights':None
    }
    args = argparse.Namespace(**dic)
    main(args)