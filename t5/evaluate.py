
import argparse

import pandas as pd

import load_t5 as t5
from python_transformers import (MODEL_WITH_LM_HEAD_MAPPING, PreTrainedTokenizer, Trainer, set_seed)
import load_data as data
import torch
import torch.nn.functional as F
import load_flan_t5 as flan_t5
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
# from torchtext.data.metrics import bleu_score
# from torchmetrics import BLEUScore
from nltk.translate.bleu_score import sentence_bleu


def text_from_latent_code(model, input_ids, start=None):
    past = model.get_hidden(input_ids)
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
        decoder_tokenizer=model.tokenizer
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


def sample_sequence_conditional(model, length, context, past=None, num_samples=1, temperature=1, top_k=0, top_p=0.0, device='cpu', decoder_tokenizer=None, cvae=False):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    i=0
    with torch.no_grad():
        while i<length:
            inputs = {'input_ids': generated, 'encoder_hidden_states': past}
            sequence_output = model.decoder(**inputs)[0]  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            sequence_output = sequence_output * (model.model_dim ** -0.5)
            outputs = model.lm_head(sequence_output)
            next_token_logits = outputs[0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)

            if next_token.unsqueeze(0)[0,0].item() == 1:
                    break

            i+=1

    return generated


def inference_given_text(p1, p2, type=None, model=None, injection_type="encoder_prefix"):
    if injection_type == 'decoder_prefix':
        t0 = [i for i in p1.split(" ") if i not in (" ", ',')]
        t1 = [i for i in p2.split(" ") if i not in (" ", ',')]
        text = ' '.join(t0) + ' </s> ' + ' '.join(t1) + ' </s>'
        inp = model.tokenizer.batch_encode_plus([text], max_length=70, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
        input_ids = inp['input_ids']
        start = '</s> '+'the inference type is '+type if type != None else '</s>'
    elif injection_type == 'encoder_prefix':
        t0 = [i for i in p1.split(" ") if i not in (" ", ',')]
        t1 = [i for i in p2.split(" ") if i not in (" ", ',')]
        text = 'inference type is '+type+' </s> '+' '.join(t0) + ' </s> ' + ' '.join(t1) + ' </s>'
        inp = model.tokenizer.batch_encode_plus([text], max_length=70, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
        input_ids = inp['input_ids']
        start = '</s>'
    else:
        exit()

    print("pred conclusion: ", text_from_latent_code(model, input_ids, start=start))


def flan_T5_inference_given_text(text, type=None, model=None):
    inp = model.tokenizer.batch_encode_plus([text], max_length=70, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
    input_ids = inp['input_ids']
    start = '</s> '+'the inference type is '+type if type != None else '</s>'

    print("pred conclusion: ", text_from_latent_code(model, input_ids, start=start))


if __name__ == '__main__':
    """
    latent_type
    latent_vec
    pretrain_model_path
    train_data_file
    inject_way
    """
    model_dict = {'model_path': '', 't5_model_name': 't5-base', 'model_type': 't5', 'config_name': None,
                  'tokenizer_name': None, 'cache_dir': None, 'ae_latent_size': 1000, 'set_seq_size': 70,
                  'latent_vec': 'pooling', 'latent_type':'T5_original', 'latent_vec_dec': 'linear',
                  'pretrain_model_path':'checkpoints/checkpoints_with_inference_type_decoder_end/t5_original_loss_0.49/train'}

    data_dict = {'train_data_file': 'datasets/full/tr_data_with_fine_type.csv',
                 'test_data_file': 'datasets/full/te_data_with_fine_type.csv',
                 'overwrite_cache': True,
                 'inject_way': 'decoder_end'}

    # parameter illustration:
    # output_dir: output dir for saving model
    # num_train_epochs: training epoch
    # per_device_train_batch_size: batch_size

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

    model_args = argparse.Namespace(**model_dict)
    data_args = argparse.Namespace(**data_dict)
    training_args = argparse.Namespace(**training_dict)

    args = {}
    args.update(model_dict)
    args.update(data_dict)
    args.update(training_dict)
    args = argparse.Namespace(**args)

    # check here before training.
    args.load_pretrain_t5 = True
    args.device = 'cpu'
    args.beta = 1.0
    args.log_interval = 10
    set_seed(training_args.seed)

    # checking latent connection mode.
    if model_args.latent_type == 'T5_original':
        if model_args.t5_model_name == 't5-base':
            model = t5.load_t5_origin(model_args, training_args) if args.load_pretrain_t5 else t5.new_t5_origin(model_args, training_args)
        elif model_args.t5_model_name == "google/flan-t5-base":
            model = flan_t5.load_flan_t5_original(model_args, training_args) if args.load_pretrain_t5 else flan_t5.new_flan_t5_original(model_args, training_args)
        else:
            exit('Error: wrong model name (two options: t5-base or google/flan-t5-base)')
    elif model_args.latent_type == 'T5_vae':
        model = t5.load_t5_vae(model_args, training_args) if args.load_pretrain_t5 else t5.new_t5_vae(model_args, training_args)
    else:
        model = t5.load_t5_ae(model_args, training_args) if args.load_pretrain_t5 else t5.new_t5_ae(model_args, training_args)

    model.to(args.device)



    # bleurt score
    bleurt_tokenizer = AutoTokenizer.from_pretrained("Elron/bleurt-base-512")
    bleurt_model = AutoModelForSequenceClassification.from_pretrained("Elron/bleurt-base-512")
    bleurt_model.eval()

    # sentenceT5 similarity
    sentenceT5_model = SentenceTransformer('sentence-transformers/sentence-t5-base')

    # control inference type to generate conclusion.
    # p1 = "seeing requires light"
    # p2 = "reading requires seeing"
    # c = "the inference type is further_specification volcano eruption is a quick process to form land"
    # for t in ['arg_substitution', 'verb_substitution', 'substitution', 'inference_from_rule', 'infer_class_from_properties', 'further_specification', 'conjunction', 'init', 'if_then', 'example']:
    #     print(t)
    #     inference_given_text(p1, p2, t, model)
    #
    # exit()

    # generate test set conclusion.
    train_dataset, test_dataset = data.get_dataset(data_args, tokenizer=model.tokenizer,
                                                   set_seq_size=model_args.set_seq_size,
                                                   local_rank=training_args.local_rank,
                                                   inject=args.inject_way)

    test_dataloader = data.get_dataloader(args, test_dataset)
    train_dataloader = data.get_dataloader(args, train_dataset)

    index = 0
    scores_sum_cos, scores_sum_bleurt, scores_sum_bleu = 0, 0, 0
    acc = 0

    with torch.no_grad():
        for step, inputs in enumerate(test_dataloader):

            input, label = inputs['input_ids'], inputs['label_ids']
            input_ids = input.to(args.device)
            output_ids = label.to(args.device)

            conclusion = model.tokenizer.decode(output_ids.tolist()[0][1:])

            # ---------------------------------- decoder prefix ----------------------------------
            # gold_type = conclusion.split(' ')[4]
            # gold_con = ' '.join(conclusion.split(' ')[5:])
            #
            # pred_conclusion = text_from_latent_code(model, input_ids)
            # pred_type = pred_conclusion.split(' ')[4]
            # pred_con = ' '.join(pred_conclusion.split(' ')[5:])

            # ---------------------------------- decoder endfix ----------------------------------
            tmp = conclusion.split(' the inference type is ')
            gold_con, gold_type = tmp[0], tmp[1]
            tmp1 = text_from_latent_code(model, input_ids).split(' the inference type is ')
            pred_con, pred_type = tmp1[0], tmp1[1]

            # ----------------- encoder prefix AND without Inference Type-------------------------
            # gold_con = conclusion
            # pred_con = text_from_latent_code(model, input_ids)
            # ------------------------------------------------------------------------------------

            print(str(step)+'##'*10)
            premises = model.tokenizer.decode(input_ids.tolist()[0])
            print("premises: ", premises)
            print("gold con: ", gold_con)
            print("pred con: ", pred_con)
            print("gold type: ", gold_type)
            print("pred type: ", pred_type)

            # -------------------------------------- Inference Type Prediction ---------------------------------------
            # pred_type = text_from_latent_code(model, input_ids, start='</s> '+gold_con+' the inference type is ').split('the inference type is')[1]
            # print("pred type: ", pred_type.strip())
            # print("gold type: ", gold_type.strip())
            #
            # if 'substitution' in pred_type.strip() and 'substitution' in gold_type.strip():
            #     acc += 1
            # elif pred_type.strip() == gold_type.strip():
            #     acc += 1
            # else:
            #     pass


            # # --------------------------------------------- BLEURT -------------------------------------------------
            # references = [gold_con]
            # candidates = [pred_con]
            # with torch.no_grad():
            #     bleurt_scores = bleurt_model(**bleurt_tokenizer(references, candidates, return_tensors='pt'))[0].squeeze().item()

            # if bleurt_scores < 0.5:
            #     print("premises: ", premises)
            #     print("gold con: ", gold_con)
            #     print("pred con: ", pred_con)
            #     print(bleurt_scores)

            # # ------------------------------------------- SentenceT5 -----------------------------------------------
            # sentences = [pred_con, gold_con]
            # embeddings = sentenceT5_model.encode(sentences)
            # embed1 = torch.FloatTensor(embeddings[0])
            # embed2 = torch.FloatTensor(embeddings[1])
            # cos_scores = torch.cosine_similarity(embed1, embed2, dim=0)
            #
            # # ---------------------------------------------- BLEU --------------------------------------------------
            # references = [gold_con.split(' ')]
            # candidates = pred_con.split(' ')
            # bleu_scores = sentence_bleu(references, candidates, weights=(1, 0, 0, 0))
            #
            index += 1
            # scores_sum_cos += cos_scores
            # scores_sum_bleu += bleu_scores
            # scores_sum_bleurt += bleurt_scores


    # print("bleu: ", scores_sum_bleu/index)
    # print("bleurt: ", scores_sum_bleurt/index)
    # print("cosine: ", scores_sum_cos/index)
    # print(acc/index)
