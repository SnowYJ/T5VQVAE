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
# import umap
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
# from torchtext.data.metrics import bleu_score
# from torchmetrics import BLEUScore
from nltk.translate.bleu_score import sentence_bleu
import umap.umap_ as umap
from yellowbrick.text import TSNEVisualizer
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm


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


def traversal(p1, p2, dim, top_k):
    t0 = [i for i in p1.split(" ") if i not in (" ", ',')]
    input = model_t5.tokenizer.batch_encode_plus([' '.join(t0)], max_length=args.set_seq_size, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')['input_ids'][0].unsqueeze(0)

    t1 = [i for i in p2.split(" ") if i not in (" ", ',')]
    input1 = model_t5.tokenizer.batch_encode_plus([' '.join(t1)], max_length=args.set_seq_size, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')['input_ids'][0].unsqueeze(0)

    input_embed = model_t5.get_hidden(input)
    input_embed1 = model_t5.get_hidden(input1)
    quantized1 = model_t5.vqvae(input_embed1)[1]

    input_shape = input_embed.shape
    distances = model_t5.vqvae.get_latent(input_embed)
    # choose the minimal index.
    encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
    # print(encoding_indices)

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
        encoding01 = torch.cat((quantized, quantized1), 1)
        attention_mask = torch.cat((input.ne(model_t5.config.pad_token_id).long(), input1.ne(model_t5.config.pad_token_id).long()), 1)
        quantized = model_t5.inference_model.encoder(inputs_embeds=encoding01, attention_mask=attention_mask)[0]

        pred_con = text_from_latent_code(model_t5, quantized, args=args)

        print("dim {} sent {}: {} ".format(dim, n, pred_con))


if __name__ == '__main__':

    # --------------------------------------------only use to build T5--------------------------------------------------
    # pretrain_model_path: loading the pretrained T5.
    # output_dir
    # train_data_file: tr_data.csv, te_data.csv
    # test_data_file
    # inject_way: if using the datasets with inference types, you need to specific the inject_way.
    # per_device_train_batch_size
    # per_device_eval_batch_size

    # model_dict = {'model_path': '', 't5_model_name': "t5-base", 'model_type': 't5', 'config_name': None,
    #               'tokenizer_name': None, 'cache_dir': None, 'ae_latent_size': 768, 'set_seq_size': 45,
    #               'latent_vec': None,
    #               'latent_vec_dec': None,
    #               'latent_type':'T5_vqvae',
    #               'code_book_size': 10000,
    #               'latent_num_layer': None,
    #               'two_stage_vqvae': False,
    #               'pretrain_model_path':'checkpoints/vqvae/t5_base_inf_sep_vqvae_pretrained_inference_t5base_loss_0_88_45_10000/train'}

    model_dict = {'model_path': '', 't5_model_name': "t5-base", 'model_type': 't5', 'config_name': None,
                  'tokenizer_name': None, 'cache_dir': None, 'ae_latent_size': 768, 'set_seq_size': 80,
                  'latent_vec': None,
                  'latent_vec_dec': None,
                  'latent_type':'T5_vqvae',
                  'latent_num_layer': None,
                  'two_stage_vqvae': False,
                  'disentangled_vqvae': False,
                  'code_book_size': 10000,
                  'hard_code_select': False,
                  'pretrain_model_path':'checkpoints/vqvae/t5_base_inf_vqvae_loss_0.85_80_10000/train'}

    data_dict = {'train_data_file': 'datasets/full/tr_data.csv',
                 'test_data_file': 'datasets/full/te_data.csv',
                 'overwrite_cache': True,
                 'task': 'inference_sep',
                 'inject_way': 'encoder_prefix'}

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
    model_vqvae = VectorQuantizerEMA(num_embeddings=model_args.code_book_size, embedding_dim=model_args.ae_latent_size, commitment_cost=0.25, decay=0.99, disentanglement=model_args.disentangled_vqvae, **dimension_dict)
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

    # ----------------------------------------------- Traversal --------------------------------------------------------
    # premise1 = "a fish is a kind of aquatic animal"
    # premise2 = "a shark is a kind of fish"
    # gold = "sharks are a kind of aquatic animal"
    # pred = "a shark is a kind of aquatic animal"
    # # for i in range(10):
    # traversal(premise2, premise1, dim=2, top_k=20)
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
        for step, inputs in enumerate(train_dataloader):
            input, output, label, input1 = inputs['input_ids'], inputs['label_ids'], inputs['label1_ids'], inputs['input1_ids']

            premise1 = model_t5.get_hidden(input)
            premise2 = model_t5.get_hidden(input1)
            encoding0 = model_t5.vqvae(premise1)[1] # p1
            encoding1 = model_t5.vqvae(premise2)[1] # p2

            encoding01 = torch.cat((encoding0, encoding1), 1)
            attention_mask = torch.cat((input.ne(model_t5.config.pad_token_id).long(), input1.ne(model_t5.config.pad_token_id).long()), 1)
            quantized = model_t5.inference_model.encoder(inputs_embeds=encoding01, attention_mask=attention_mask)[0]

            pred_con = text_from_latent_code(model_t5, quantized, args=args)
            gold_con = model_t5.tokenizer.decode(label.tolist()[0])
            print('############')
            print("premise1: ", model_t5.tokenizer.decode(input.tolist()[0]))
            print("premise2: ", model_t5.tokenizer.decode(input1.tolist()[0]))
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
