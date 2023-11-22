import t5.load_data as data
import argparse
import t5.load_t5 as t5
from train_dvae import build_model
from dvae.model import build_VAE, build_DKF, build_STORN, build_VRNN, build_SRNN, build_RVAE, build_DSAE
from dvae.utils import myconf, get_logger, loss_ISD, loss_KLD, loss_MPJPE, loss_MSE
import torch
from t5.utils import frange_cycle_zero_linear
import os
import torch.nn.functional as F
from vqvae.vqvae import VectorQuantizer, VectorQuantizerEMA
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu
from allennlp.predictors.predictor import Predictor
from nltk.corpus import stopwords
from nltk import download
import gensim.downloader as api
import random


def text_from_latent_code(model, input_ids, start=None, args=None):
    if args.latent_type == 'T5_original':
        past = model.get_hidden(input_ids)
    elif args.latent_type == 'T5_vqvae':
        # attention_mask = input_ids.ne(model_t5.config.pad_token_id).long()
        # encoding = model_t5.t5_model.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        # past = model.vqvae.get_hidden(encoding)
        past = input_ids
    else:
        past = None
        exit("ERROR")

    context_tokens = model.tokenizer.encode('</s>') if start == None else model.tokenizer.encode(start)
    args.top_k = 0
    args.top_p = 1.0
    args.temperature = 1
    length = 40

    out = sample_sequence_conditional(
        model=model.t5_model,
        context=context_tokens,
        past=past,
        length=length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=args.device,
        decoder_tokenizer=model.tokenizer,
        args=args
    )

    text_x1 = model.tokenizer.decode(out[0,:].tolist()) # , clean_up_tokenization_spaces=True
    text_x1 = text_x1.split()
    text_x1 = ' '.join(text_x1)

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


def sample_sequence_conditional(model, length, context, past=None, num_samples=1, temperature=1, top_k=0, top_p=0.0, device='cpu', decoder_tokenizer=None, cvae=False, args=None):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    i=0
    with torch.no_grad():
        while i<length:
            inputs = {'input_ids': generated, 'encoder_hidden_states': past}
            if args.model_type == 't5':
                sequence_output = model.decoder(**inputs)[0]  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
                sequence_output = sequence_output * (model.model_dim ** -0.5)
            elif args.model_type == 'bart':
                sequence_output = model.model.decoder(**inputs)[0]  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
                sequence_output = sequence_output * (768 ** -0.5)
            else:
                exit()

            outputs = model.lm_head(sequence_output)
            next_token_logits = outputs[0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)

            # print(next_token.unsqueeze(0)[0,0].item())
            if next_token.unsqueeze(0)[0,0].item() == 1:
                break

            i+=1

    return generated


def traversal(input_text, dim, top_k):
    t0 = [i for i in input_text.split(" ") if i not in (" ", ',')]
    input = model_t5.tokenizer.batch_encode_plus([' '.join(t0)], max_length=args.set_seq_size, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')['input_ids'][0].unsqueeze(0)
    # input_embed = torch.tensor(model_t5.tokenizer.encode(' '.join(t0))).unsqueeze(0)
    input_embed = model_t5.get_hidden(input)
    input_shape = input_embed.shape
    distances = model_t5.vqvae.get_latent(input_embed)
    # choose the minimal index.
    encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

    # choose the top K minimal value for dim.
    a, idx1 = torch.sort(distances[dim], descending=False) # descending为 alse，升序，为True，降序
    idx = idx1[:top_k]

    for n, i in enumerate(idx):
        # replace
        encoding_indices[dim][0] = i
        encodings = torch.zeros(encoding_indices.shape[0], model_t5.vqvae._num_embeddings, device=args.device)
        encodings.scatter_(1, encoding_indices, 1)
        # Quantize and unflatten
        quantized = torch.matmul(encodings, model_t5.vqvae._embedding.weight).view(input_shape)

        # generated_ids = model_t5.t5_model.generate(
        #     input_ids=input,
        #     attention_mask=input != 1,
        #     max_length=40,
        #     num_beams=2,
        #     repetition_penalty=5,
        #     length_penalty=1.0,
        #     early_stopping=True,
        #     encoder_embed=quantized
        # )
        # pred_con = [model_t5.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids][0].strip()
        pred_con = text_from_latent_code(model_t5, quantized, args=args)

        print("dim {} sent {}: {} ".format(dim, n, pred_con))


def interpolation(source, target, way='euclidean'):
    interpolation_list = []
    t0 = [i for i in source.split(" ") if i not in (" ", ',')]
    source = model_t5.tokenizer.batch_encode_plus([' '.join(t0)], max_length=args.set_seq_size, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')['input_ids'][0].unsqueeze(0)
    source_embed = model_t5.get_hidden(source)

    t1 = [i for i in target.split(" ") if i not in (" ", ',')]
    target = model_t5.tokenizer.batch_encode_plus([' '.join(t1)], max_length=args.set_seq_size, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')['input_ids'][0].unsqueeze(0)
    target_embed = model_t5.get_hidden(target)

    if way == 'euclidean':
        # operate the encoder embedding directly.
        for z in range(0, 11, 1):
            z_t = z/10
            current_embed = z_t * target_embed + (1-z_t) * source_embed
            current_code = model_t5.vqvae(current_embed)[1]
            current_output = text_from_latent_code(model_t5, current_code, args=args)
            interpolation_list.append(current_output)
            print(current_output)
    else:
        # operate the VQ latent space directly.
        source_code = model_t5.vqvae(source_embed)[1].squeeze(0)
        target_code = model_t5.vqvae(target_embed)[1].squeeze(0)

        for z in range(0, 11, 1):
            source_t = 1 - z/10
            target_t = z/10
            # calculate the distance between source_code and all codes.
            left_distances = (torch.sum(source_code**2, dim=1, keepdim=True) + torch.sum(model_t5.vqvae._embedding.weight**2, dim=1) - 2 * torch.matmul(source_code, model_t5.vqvae._embedding.weight.t()))
            right_distances = (torch.sum(target_code**2, dim=1, keepdim=True) + torch.sum(model_t5.vqvae._embedding.weight**2, dim=1) - 2 * torch.matmul(target_code, model_t5.vqvae._embedding.weight.t()))
            distances = source_t * left_distances + target_t * right_distances

            # # choose the TopK closed points to source code. from them, choose the closed point to target code.
            # _, idx1 = torch.sort(left_distances, descending=False) # if z < size/2 else torch.sort(right_distances, descending=False)
            # idx = idx1[:, :top_k]
            # gathered_tensor = torch.gather(distances, 1, idx)
            # min_indices = torch.argmin(gathered_tensor, dim=1)
            # encoding_indices = torch.tensor([idx[r, c].item() for r, c in enumerate(min_indices)]).view(-1, 1)

            encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1) # 45, 1
            encodings = torch.zeros(encoding_indices.shape[0], model_t5.vqvae._num_embeddings, device=args.device)
            encodings.scatter_(1, encoding_indices, 1)
            quantized = torch.matmul(encodings, model_t5.vqvae._embedding.weight)
            source_code = quantized
            current_output = text_from_latent_code(model_t5, quantized.unsqueeze(0), args=args)
            print(current_output)
            interpolation_list.append(current_output)

    return interpolation_list


# def get_attention_weights(source):
#     t0 = [i for i in source.split(" ") if i not in (" ", ',')]
#     source = model_t5.tokenizer.batch_encode_plus([' '.join(t0)], max_length=args.set_seq_size, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')['input_ids'][0].unsqueeze(0)
#     source0_embed = model_t5.get_hidden(source)
#     current_code = model_t5.vqvae(source0_embed)[1]
#     sequence_output = model_t5.t5_model.decoder(input_ids=decoder_input_ids, encoder_hidden_states=current_code, output_attentions=True)


def arithmetic(source1, source2, source3, operator):
    t0 = [i for i in source1.split(" ") if i not in (" ", ',')]
    source = model_t5.tokenizer.batch_encode_plus([' '.join(t0)], max_length=args.set_seq_size, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')['input_ids'][0].unsqueeze(0)
    source0_embed = model_t5.get_hidden(source)
    # source1_latent = model_t5.vqvae(source_embed)[1]
    # latent_source = model_t5.vqvae(source_embed)[1]

    t1 = [i for i in source2.split(" ") if i not in (" ", ',')]
    target = model_t5.tokenizer.batch_encode_plus([' '.join(t1)], max_length=args.set_seq_size, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')['input_ids'][0].unsqueeze(0)
    source1_embed = model_t5.get_hidden(target)
    # source2_latent = model_t5.vqvae(target_embed)[1]
    # latent_target = model_t5.vqvae(target_embed)[1]

    if source3 != None:
        t2 = [i for i in source3.split(" ") if i not in (" ", ',')]
        target1 = model_t5.tokenizer.batch_encode_plus([' '.join(t2)], max_length=args.set_seq_size, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')['input_ids'][0].unsqueeze(0)
        source2_embed = model_t5.get_hidden(target1)
        current_code = source0_embed + source1_embed - source2_embed
    else:
        if operator == 'add':
            current_code = source0_embed + source1_embed
        elif operator == 'sub':
            current_code = source0_embed - source1_embed
        else:
            current_code = (source0_embed + source1_embed)/2

    # if operator == 'add':
    # elif operator == 'sub':
    #     current_code = source_embed - target_embed
    # else:
    #     current_code = (source_embed + target_embed)/2

    current_code = model_t5.vqvae(current_code)[1]
    current_output = text_from_latent_code(model_t5, current_code, args=args)
    print(current_output)


def quasi_symbolic_inference(source1, source2, w1, w2, type='substitution', right=True):
    # split the words
    w1_list = model_t5.tokenizer.convert_ids_to_tokens(model_t5.tokenizer(w1)['input_ids'])
    print(w1_list)

    # split the sentence (minor premise).
    t0 = [i for i in source1.split(" ") if i not in (" ", ',')]
    source = model_t5.tokenizer.batch_encode_plus([' '.join(t0)], max_length=args.set_seq_size, pad_to_max_length=True,
                                                  truncation=True, padding="max_length", return_tensors='pt')['input_ids'][0].unsqueeze(0)
    token_split = model_t5.tokenizer.convert_ids_to_tokens(source[0])
    print(token_split)
    # find indices from minor premises.
    indices = []
    for index, i in enumerate(token_split):
        if i == '<pad>':
            exit('ERROR: not find keyword in minor premise.')

        if w1_list == token_split[index:index+len(w1_list)]:
            indices = [k for k in range(index,index+len(w1_list))]
            break

    w1_index = indices
    source0_embed = model_t5.get_hidden(source)
    source0_quantized = model_t5.vqvae(source0_embed)[1]

    # find the code embeddings according to indices.
    w1_quantized = source0_quantized[0, w1_index]
    w1_quantized_pad = source0_quantized[0, -1].unsqueeze(0)

    # ------------------------------------------------------------------------------------------------------------------

    # same process for second premise.
    w2_list = model_t5.tokenizer.convert_ids_to_tokens(model_t5.tokenizer(w2)['input_ids'])
    print(w2_list)
    t1 = [i for i in source2.split(" ") if i not in (" ", ',')]
    target = model_t5.tokenizer.batch_encode_plus([' '.join(t1)], max_length=args.set_seq_size, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')['input_ids'][0].unsqueeze(0)
    source1_embed = model_t5.get_hidden(target)
    source1_quantized = model_t5.vqvae(source1_embed)[1]
    token_split = model_t5.tokenizer.convert_ids_to_tokens(target[0])
    print(token_split)
    indices = []
    for index, i in enumerate(token_split):
        if i == '<pad>':
            exit('ERROR: not find keyword in major premise.')

        if w2_list == token_split[index:index+len(w2_list)]:
            indices = [k for k in range(index,index+len(w2_list))]
            break

    w2_index = indices

    if type == 'substitution':
        if len(w2_index) == len(w1_index):
            source1_quantized[0, w2_index] = w1_quantized
        else:
            w2_index_left, w2_index_right = w2_index[0], w2_index[-1]
            tmp_source1_left = source1_quantized[0, :w2_index_left]
            tmp_source1_right = source1_quantized[0, w2_index_right+1:]
            tmp_source1_middle = w1_quantized

            source1_quantized = torch.cat([tmp_source1_left, tmp_source1_right, tmp_source1_middle], dim=0)

            if source1_quantized.shape[0] < args.set_seq_size:
                fix_len = args.set_seq_size - source1_quantized.shape[0]
                fix_code = source1_quantized[-1].unsqueeze(0)
                fix_code = torch.cat([fix_code]*fix_len)
                source1_quantized = torch.cat([source1_quantized, fix_code], dim=0).unsqueeze(0)
            else:
                source1_quantized = source1_quantized[:45].unsqueeze(0)

    elif type == 'further_specification':
        # c = p2 + part of p1
        w2_index_left, w2_index_right = w2_index[0], w2_index[-1]
        tmp_source1_leftmost = source1_quantized[0, :w2_index_left]
        tmp_source1_rightmost = source1_quantized[0, w2_index_right+1:]
        tmp_source1_leftmiddle = source1_quantized[0, w2_index_left:w2_index_right+1]
        tmp_source1_rightmiddle = w1_quantized # torch.cat([w1_quantized_pad, w1_quantized], dim=0)
        if right:
            # p2 p1
            source1_quantized = torch.cat([tmp_source1_leftmost, tmp_source1_leftmiddle, tmp_source1_rightmiddle, tmp_source1_rightmost], dim=0)
        else:
            # p1 p2
            source1_quantized = torch.cat([tmp_source1_leftmost, tmp_source1_rightmiddle, tmp_source1_leftmiddle, tmp_source1_rightmost], dim=0)
        source1_quantized = source1_quantized[:45].unsqueeze(0)
        # source1_quantized = torch.cat([source0_quantized, source1_quantized], dim=1)
    elif type == 'conjunction':
        # c = p2 and part of p1
        # --------------------------------------------------------------------------------------------------------------
        conjunction_and = model_t5.tokenizer.convert_ids_to_tokens(model_t5.tokenizer('and')['input_ids'])
        conjunction_and = model_t5.tokenizer.batch_encode_plus(conjunction_and, max_length=args.set_seq_size, pad_to_max_length=True,
                                                      truncation=True, padding="max_length", return_tensors='pt')['input_ids'][0].unsqueeze(0)
        and_embed = model_t5.get_hidden(conjunction_and)
        and_quantized = model_t5.vqvae(and_embed)[1][0, 0, :].view(1, -1)

        conjunction_both = model_t5.tokenizer.convert_ids_to_tokens(model_t5.tokenizer('both')['input_ids'])
        conjunction_both = model_t5.tokenizer.batch_encode_plus(conjunction_both, max_length=args.set_seq_size, pad_to_max_length=True,
                                                               truncation=True, padding="max_length", return_tensors='pt')['input_ids'][0].unsqueeze(0)
        both_embed = model_t5.get_hidden(conjunction_both)
        both_quantized = model_t5.vqvae(both_embed)[1][0, 0, :].view(1, -1)
        # --------------------------------------------------------------------------------------------------------------
        w2_index_left, w2_index_right = w2_index[0], w2_index[-1]
        tmp_source1_leftmost = source1_quantized[0, :w2_index_left]
        tmp_source1_rightmost = source1_quantized[0, w2_index_right+1:]
        tmp_source1_leftmiddle = source1_quantized[0, w2_index_left:w2_index_right+1]
        tmp_source1_rightmiddle = w1_quantized # torch.cat([w1_quantized_pad, w1_quantized], dim=0)
        source1_quantized = torch.cat([tmp_source1_leftmost, tmp_source1_leftmiddle, and_quantized, tmp_source1_rightmiddle, tmp_source1_rightmost], dim=0)
        source1_quantized = source1_quantized[:45].unsqueeze(0)

    else:
        exit()

    current_code = model_t5.vqvae(source1_quantized)[1]
    current_output = text_from_latent_code(model_t5, current_code, args=args)
    print(current_output)


def alignment(sent1, sent2, way='SRL'):
    if way == 'SRL':
        predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")
    else:
        predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz")

    # srl1 = predictor.predict(sentence=sent1)['verbs'][0]['tags']
    # srl2 = predictor.predict(sentence=sent2)['verbs'][0]['tags']
    srl1 = predictor.predict(sentence=sent1)
    srl2 = predictor.predict(sentence=sent2)
    print(srl1)
    print(srl2)


def preprocess(sentence, stop_words):
    return [w for w in sentence.lower().split() if w not in stop_words]


def word_mover_distance(sent1, sent2, model, stopword):
    sent1 = preprocess(sent1, stopword)
    sent2 = preprocess(sent2, stopword)
    distance = model.wmdistance(sent1, sent2)
    return distance


if __name__ == '__main__':

    # --------------------------------------------only use to build T5--------------------------------------------------
    # pretrain_model_path: loading the pretrained T5.
    # output_dir
    # train_data_file: tr_data.csv, te_data.csv
    # test_data_file
    # inject_way: if using the datasets with inference types, you need to specific the inject_way.
    # per_device_train_batch_size
    # per_device_eval_batch_size

    model_dict = {'model_path': '', 't5_model_name': "t5-base", 'model_type': 't5', 'config_name': None,
                  'tokenizer_name': None, 'cache_dir': None, 'ae_latent_size': 768, 'set_seq_size': 45,
                  'latent_vec': None,
                  'latent_vec_dec': None,
                  'latent_type':'T5_vqvae',
                  'code_book_size': 10000,
                  'latent_num_layer': None,
                  'disentangled_vqvae': False,
                  'hard_code_select': False,
                  'pretrain_model_path':'checkpoints/t5vqvae_base_10000_explanations_ema_epoch_100_batch_64_log_10/train'} # 5_base_rec_all_vqvae_0.037_45_10000

    data_dict = {'train_data_file': 'datasets/full/MathSymbol/both_tr_char.txt',
                 'test_data_file': 'datasets/full/MathSymbol/EVAL_differentiation_char.txt',
                 'overwrite_cache': True,
                 'inject_way': 'encoder_prefix',
                 'task': 'math_recon'} # recon: natural language, math_recon: symbolic language

    training_dict = {'output_dir': './output', 'overwrite_output_dir': True, 'do_train': True, 'do_eval': False,
                     'do_predict': False, 'evaluate_during_training': False, 'per_device_train_batch_size': 1,
                     'per_device_eval_batch_size': 1, 'per_gpu_train_batch_size': None, 'per_gpu_eval_batch_size': None,
                     'gradient_accumulation_steps': 1, 'learning_rate': 5e-05, 'weight_decay': 0.0, 'adam_epsilon': 1e-08,
                     'max_grad_norm': 1.0, 'num_train_epochs': 10, 'max_steps': -1, 'warmup_steps': 0,
                     'logging_dir': 'runs/Oct30_17-24-56_192.168.1.104', 'logging_first_step': False, 'logging_steps': -1,
                     'save_steps': -1, 'save_total_limit': 1, 'no_cuda': False, 'seed': 42, 'fp16': False, 'fp16_opt_level': 'O1',
                     'local_rank': -1, 'tpu_num_cores': None, 'tpu_metrics_debug': False, 'debug': False,
                     'dataloader_drop_last': False, 'eval_steps': 1000, 'past_index': -1, 'project_name': 'test',
                     'reg_schedule_k': 0.0025, 'reg_schedule_b': 6.25, 'reg_constant_weight': None,
                     'use_recon_loss': False}

    # ------------------------------------------------------------------------------------------------------------------

    model_args = argparse.Namespace(**model_dict)
    data_args = argparse.Namespace(**data_dict)
    training_args = argparse.Namespace(**training_dict)

    args = {}
    args.update(model_dict)
    args.update(data_dict)
    args.update(training_dict)
    args = argparse.Namespace(**args)
    args.device = 'cpu'

    # -------------------------------------------load and train dvae----------------------------------------------------
    # args.dvae_name = 'VRNN'
    # args.config_file = 'checkpoints/t5_dvae_rec_0.59/cfg_vrnn.ini'
    # args.log_interval = 10
    # args.beta = 1.0
    # args.dvae_save_path = 'checkpoints/vrnn_kl_0/train.pt'
    # ------------------------------------------------------------------------------------------------------------------

    # load dvae.
    dimension_dict = {"ARG0": (0, 2000), "ARG1": (2000, 4000), "ARG2": (4000, 6000), "V": (6000, 8000), "residual": (8000, 10000)}
    disentangled_vqvae = False
    model_vqvae = VectorQuantizerEMA(num_embeddings=model_args.code_book_size, embedding_dim=model_args.ae_latent_size, commitment_cost=0.25, decay=0.99, disentanglement=disentangled_vqvae, **dimension_dict)
    # model_vqvae = VectorQuantizerEMA(num_embeddings=10000, embedding_dim=512, commitment_cost=0.25, decay=0.99)


    # load pretrain T5.
    # model_t5 = t5.load_t5_origin(model_args, training_args)
    model_t5 = t5.load_t5_vqvae(args, training_args, model_vqvae, load_model="T5-vqvae")
    model_t5.to(args.device)

    # bleurt score
    bleurt_tokenizer = AutoTokenizer.from_pretrained("Elron/bleurt-base-512")
    bleurt_model = AutoModelForSequenceClassification.from_pretrained("Elron/bleurt-base-512")
    bleurt_model.eval()

    # sentenceT5 similarity
    sentenceT5_model = SentenceTransformer('sentence-transformers/sentence-t5-base')

    # ---------------------------------------------- heatmap ----------------------------------------------



    # -----------------------------------------------------------------------------------------------------

    # ------------------------------------------------------ visualization ---------------------------------------------

    # X = np.array(model_t5.vqvae._embedding.weight.data)
    # np.save("latent_space_analysis/codebook.npy", X)
    # exit()
    # kmeans = KMeans(n_clusters=3, random_state=0, n_init=500).fit(X)
    # label = kmeans.labels_
    # # np.savetxt("label.npy", label)
    # print(np.load("latent_space_analysis/order_latent.npy").shape)
    # exit()
    # train_dataset, test_dataset = data.get_dataset(data_args, tokenizer=model_t5.tokenizer, set_seq_size=model_args.set_seq_size, disentangle=True)
    # train_dataloader = data.get_dataloader(args, train_dataset)
    # test_dataloader = data.get_dataloader(args, test_dataset)
    # order_index, order_token, order_role, order_latent, order_text = [], [], [], [], []
    # with torch.no_grad():
    #     for (step, inputs) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
    #         input, output, label, srl = inputs['input_ids'], inputs['label_ids'], inputs['label1_ids'], inputs['srl']
    #         input_embed = model_t5.get_hidden(input)
    #         ind = model_t5.vqvae.get_indices(input_embed, srl=None)
    #         txt = model_t5.tokenizer.convert_ids_to_tokens(input.tolist()[0])
    #
    #         r = [i[0] for i in srl]
    #         index = list(np.array(ind).reshape(1, -1)[0]) # [:len(r)]
    #         text = ' '.join(txt[:len(r)])
    #         role = ' '.join(r)
    #         vector = input_embed.numpy() # [:, :len(r), :]
    #         exp = model_t5.tokenizer.decode(input.tolist()[0])
    #
    #         order_index.append(index)
    #         order_role.append(role)
    #         order_token.append(text)
    #         order_latent.append(vector)
    #         order_text.append(exp)
    #
    # import pickle
    # # Save dictionary to a file
    #
    # np.save('latent_space_analysis/order_index.npy', order_index)
    # np.save('latent_space_analysis/order_latent.npy', order_latent)
    #
    # with open('latent_space_analysis/order_token.txt', 'w') as file:
    #     for item in order_token:
    #         file.write(item + ' \n')
    #
    # with open('latent_space_analysis/order_text.txt', 'w') as file:
    #     for item in order_text:
    #         file.write(item + ' \n')
    #
    # with open('latent_space_analysis/order_role.txt', 'w') as file:
    #     for item in order_role:
    #         file.write(item + ' \n')
    # # # Load dictionary from a file
    # # with open("dict.pickle", "rb") as handle:
    # #     loaded_dict = pickle.load(handle)
    # # print(loaded_dict)
    # exit()


    # ----------------------------------------------- Traversal --------------------------------------------------------

    # exp = "travel means to move" # moons orbit planets,jupiter is a kind of planet,moons orbit jupiter
    # for i in range(10):
    #     print('#############')
    #     traversal(exp, dim=i, top_k=30)
    # exit()

    # ----------------------------------------------- interpolation ----------------------------------------------------

    # source = "solar energy transfer is a kind of interaction"
    # target = "an astronaut requires the oxygen in a spacesuit backpack to breathe"
    # interpolation(source, target)
    # exit()

    # ------------------------------------------------ Arithmetic ------------------------------------------------------

    # source1 = "the milky way is a kind of galaxy"
    # source2 = "a rock is usually a solid" # ,exposure to heat and pressure are kinds of conditions
    # arithmetic(source1, source2, None, operator='add')
    # exit()

    # ------------------------------------------------ quasi-symbolic --------------------------------------------------
    # source1 = "fungi can be multicellular"
    # source2 = "fungi have no chlorophyll" # fungi have no chlorophyll and fungi can be multicellular
    # quasi_symbolic_inference(source1, source2, "fungi can be multicellular", "chlorophyll", type='conjunction', right=True)
    # exit()

    # ------------------------------------------ interpolation smoothness ----------------------------------------------

    # model_wmd = api.load('word2vec-google-news-300')
    # download('stopwords')  # Download stopwords list.
    # stop_words = stopwords.words('english')
    # with open('./datasets/full/explanations_tr.txt') as f:
    #     lines = f.readlines()
    #
    # data = [i.split('&')[0].strip() for i in lines[:100]]
    # random.shuffle(data)
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
    #     interpolate_path = interpolation(source, target)
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

    # ------------------------------------------------ test rec --------------------------------------------------------
    # load dataset.
    train_dataset, test_dataset = data.get_dataset(data_args, tokenizer=model_t5.tokenizer, set_seq_size=model_args.set_seq_size, disentangle=dimension_dict, task=args.task)
    train_dataloader = data.get_dataloader(args, train_dataset)
    test_dataloader = data.get_dataloader(args, test_dataset)

    index = 0
    scores_sum_cos, scores_sum_bleurt, scores_sum_bleu = 0, 0, 0
    acc = 0

    with torch.no_grad():
        for step, inputs in enumerate(test_dataloader):
            input, output, label = inputs['input_ids'], inputs['label_ids'], inputs['label1_ids']

            input_embed = model_t5.get_hidden(input)
            input_shape = input_embed.shape
            distances = model_t5.vqvae.get_latent(input_embed)
            encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
            encodings = torch.zeros(encoding_indices.shape[0], model_t5.vqvae._num_embeddings, device=args.device)
            encodings.scatter_(1, encoding_indices, 1)
            # Quantize and unflatten
            quantized = torch.matmul(encodings, model_t5.vqvae._embedding.weight).view(input_shape)

            # generated_ids = model_t5.t5_model.generate(
            #     input_ids=input,
            #     attention_mask=input != 1,
            #     max_length=40,
            #     num_beams=2,
            #     repetition_penalty=5,
            #     length_penalty=1.0,
            #     early_stopping=True,
            #     encoder_embed=quantized
            # )
            # pred_con = [model_t5.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids][0].strip()

            if args.task in ["math_recon", "math_inf"]:
                pred_con = text_from_latent_code(model_t5, quantized, args=args).replace("[SEP]", "")
                gold_con = model_t5.tokenizer.decode(label.tolist()[0]).replace("[SEP]", "")
            print('############')
            print("gold: ", gold_con)
            print("pred: ", pred_con)

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

            if bleurt_scores < 0.5:
                print(bleurt_scores)
                print(bleu_scores)
                print("gold: ", gold_con)
                print("pred: ", pred_con)


            index += 1
            scores_sum_cos += cos_scores
            scores_sum_bleu += bleu_scores
            scores_sum_bleurt += bleurt_scores

    print("bleu: ", scores_sum_bleu/index)
    print("bleurt: ", scores_sum_bleurt/index)
    print("cosine: ", scores_sum_cos/index)

    # ------------------------------------------------ T-SNE plot ------------------------------------------------------

    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111)
    # tsne_vis = TSNEVisualizer(ax=ax, decompose_by=4, decompose='svd') # , classes=[content1, content2]
    # tsne_vis.fit(model_t5.vqvae._embedding.weight.data.cpu()) # , y=targets
    # tsne_vis.show()
    # fig.savefig(f"tsne_animal_human_sup_svd.png")

    # import numpy as np
    # import umap
    # import matplotlib.pyplot as plt
    #
    # # Generate sample data
    # np.random.seed(0)
    # data = model_t5.vqvae._embedding.weight.data.cpu()
    #
    # # Perform UMAP dimensionality reduction
    # reducer = umap.UMAP(n_neighbors=5, random_state=0)
    # embedding = reducer.fit_transform(data)
    #
    # # Plot the reduced data
    # plt.scatter(embedding[:, 0], embedding[:, 1], c='blue')
    # plt.show()

    # exit()
