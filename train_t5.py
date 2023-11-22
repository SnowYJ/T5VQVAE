"""
T5VQVAE
"""
import os
import torch
from python_transformers import set_seed
from python_transformers.optimization import AdamW, get_linear_schedule_with_warmup
import t5.load_t5 as t5
import t5.load_flan_t5 as flan_t5
import t5.load_data as data
import argparse
from t5.utils import frange_cycle_zero_linear, create_srl_vocab
from train_dvae import build_model
from vqvae.vqvae import VectorQuantizer, VectorQuantizerEMA
import pickle
import logging


def get_optimizers(args, model, num_training_steps):
    """
        Setup the optimizer and the learning rate scheduler, modified for when training with a VAE with an input-decoder.
    """
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps)

    return optimizer, scheduler


def _run_training_step(args, model, input, output, label, beta, srl=None, input1=None, srl_ids=None):
    input_ids = input.to(args.device)
    output_ids = output.to(args.device)
    label_ids = label.to(args.device)

    if srl is not None:
        srl = srl.to(args.device)

    if srl_ids is not None:
        srl_ids = srl_ids.to(args.device)

    if input1 is not None:
        input1 = input1.to(args.device)

    decoder_ce, loss_recon, _, loss_KL = model(input_ids, output_ids, label_ids, srl_ids=srl, input1_ids=input1,
                                               latent_num_layer=args.latent_num_layer, two_stage_vqvae=args.two_stage_vqvae, alpha=beta, srl_embed_ids=srl_ids)

    cross_entropy_loss, kl_loss, recon_loss = decoder_ce, loss_KL, loss_recon

    if args.latent_type == 'T5_dvae':
        seq_len, bs, _ = model.dvae.z_mean.shape
        kl_loss = kl_loss / (seq_len * bs)

    loss = cross_entropy_loss + beta*kl_loss + recon_loss # (this is the latent reconstruction loss in AE or VAE)

    if args.gradient_accumulation_steps > 1:
        loss = loss / args.gradient_accumulation_steps

    loss.backward()

    return loss, cross_entropy_loss, kl_loss, recon_loss


def test(args, model, test_dataloader, cur_beta, disentangled_vqvae=None):
    tot_loss = 0
    with torch.no_grad():
        for step, inputs in enumerate(test_dataloader):

            input, output, label = inputs['input_ids'], inputs['label_ids'], inputs['label1_ids']

            input = input.to(args.device)
            output = output.to(args.device)
            label = label.to(args.device)

            # if args.latent_type == 'T5_vqvae':
            #     model.vqvae._commitment_cost = cur_beta

            if disentangled_vqvae:
                srl = inputs['srl'].to(args.device)
                srl_ids = inputs['srl_ids'].to(args.device)
            else:
                srl, srl_ids = None, None

            # srl_ids = inputs['srl_ids'].to(args.device)

            input1 = inputs['input1_ids'].to(args.device) if args.task == 'inference_sep' else None # used in separate premises in inference task.

            decoder_ce, loss_recon, _, loss_KL = model(input, output, label, srl_ids=srl, input1_ids=input1,
                                                       latent_num_layer=args.latent_num_layer, two_stage_vqvae=args.two_stage_vqvae, alpha=cur_beta, srl_embed_ids=srl_ids)
            cross_entropy_loss, kl_loss, recon_loss = decoder_ce, loss_KL, loss_recon

            if args.latent_type == 'T5_dvae':
                seq_len, bs, _ = model.dvae.z_mean.shape
                kl_loss = kl_loss / (seq_len * bs)

            loss = cross_entropy_loss + cur_beta*kl_loss + recon_loss # (this is the reconstruction loss in AE or VAE)
            tot_loss += cross_entropy_loss
            args.logger.info('| epoch {:3d} | {:5d}/{:5d} batches | test loss {:.5f} , rec {:.5f} , kl {:.5f} , latent rec {:.5f}'.format(1, step+1, len(test_dataloader), loss, cross_entropy_loss, kl_loss, recon_loss))

    return tot_loss/(step+1)


def train(args, model, train_dataloader, test_dataloader, disentangled_vqvae=None):
    len_train_dataloader = len(train_dataloader)
    t_total = int(len_train_dataloader // args.gradient_accumulation_steps * args.num_train_epochs)
    num_train_epochs = args.num_train_epochs

    optimizer, scheduler = get_optimizers(args, model=model, num_training_steps=t_total)

    # Train!
    print("***** Running training *****")
    print("  Training model = ", args.latent_type)
    print("  Sentence embedding = ", args.latent_vec)
    print("  Connection decoder = ", args.latent_vec_dec)
    print("  Num Epochs = ", num_train_epochs)
    print("  batch size per device = ", args.per_device_train_batch_size)
    print("  Gradient Accumulation steps = ", args.gradient_accumulation_steps)
    print("  Total optimization steps = ", t_total)
    global_step = 0
    min_loss = 10000

    model.zero_grad()
    n_iter = int(args.num_train_epochs) * len(train_dataloader)
    beta_t_list = frange_cycle_zero_linear(n_iter, start=0.5, stop=args.beta,  n_cycle=1, ratio_increase=0.25, ratio_zero=0.25)

    # if the model is T5VAE or T5AE, fix the parameters of T5 at epoch 1.
    trigger = args.latent_type != 'T5_original'
    cur_beta = 0

    for epoch in range(args.num_train_epochs):
        model.train()

        # ---------------------fix or update parameter----------------------------
        # if epoch < 100 and trigger:
        #     for l, param in model.named_parameters():
        #         if l.startswith('inference_model'):
        #             param.requires_grad = True
        #         else:
        #             param.requires_grad = False
        # ------------------------------------------------------------------------

        args.logger.info('----'*30)
        for step, inputs in enumerate(train_dataloader):
            input, output, label = inputs['input_ids'], inputs['label_ids'], inputs['label1_ids']

            if disentangled_vqvae:
                srl, srl_ids = inputs['srl'], inputs['srl_ids'] # # used in different codebook and srl embedding.
            else:
                srl, srl_ids = None, None

            # srl_ids = inputs['srl_ids']

            input1 = inputs['input1_ids'] if args.task == 'inference_sep' else None # used in separate premises in inference task.
            cur_beta = beta_t_list[step + epoch*len_train_dataloader]

            current_loss, cn, kl, latent_rec = _run_training_step(args, model, input, output, label, 0.7, srl, input1, srl_ids)

            if (step + 1) % args.log_interval == 0:
                args.logger.info('| epoch {:3d} | {:5d}/{:5d} batches | train loss {:.5f} , rec {:.5f} , kl {:.5f} , latent rec {:.5f}'.format(epoch + 1, step, len_train_dataloader, current_loss, cn, kl, latent_rec))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

        te_loss = test(args, model, test_dataloader, cur_beta, disentangled_vqvae)

        if te_loss < min_loss:
            min_loss = te_loss
            args.logger.info('| epoch {:3d} end | avg test loss {:.5f}| saving model!'.format(epoch + 1, te_loss))
            # Save model checkpoint
            output_dir = args.output_dir
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            torch.save(model.state_dict(), os.path.join(output_dir, "train"))
            model.tokenizer.save_pretrained(args.output_dir)
            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        else:
            args.logger.info('| epoch {:3d} end | avg test loss {:.5f}| not saving model!'.format(epoch + 1, te_loss))


def main():
    # parameter illustration:
    # most parameters are default from T5.

    """
    ----------------------------------------------model parameter-------------------------------------------------------
    ae_latent_size: latent vector size. e.g., 1000.

    set_seq_size: length of the sequence. e.g., 70 or 80. # 44 in inference_sep, 70 in inference_com, 26 in recon.

    latent_vec: way for getting and connecting sentence-level representation with decoder,
    (1): pooling, shrinking, attention, sentenceT5
    (2): xxxx (reconstruct token embeddings), xxxx_as_mem (construct KV), xxxx_as_input (construct input), xxxx_as_output (construct output)

    latent_vec_dec: way for constructing decoder input.
    (1) linear
    (2) shrinking

    latent_type: way for connecting encoder-decoder.
    (1) T5_vae, (2) T5_ae, (3) T5_original

    t5_model_name: google/flan-t5-base or t5-base

    ---------------------------------------------training parameter-----------------------------------------------------

    output_dir: output dir for saving model
    num_train_epochs: training epoch
    per_device_train_batch_size: batch_size
    per_device_eval_batch_size: batch_size

    --------------------------------------------------others------------------------------------------------------------
    args.load_pretrain_t5 = True
    args.inject_way = (1) encoder_prefix, (2) decoder_prefix, (3) decoder_end

    """

    model_dict = {'model_path': '', 't5_model_name': "t5-small", 'model_type': 't5', 'config_name': None,
                  'tokenizer_name': None, 'cache_dir': None, 'ae_latent_size': 256, 'set_seq_size': 130, # 45 for recon or math_recon, 130 for math_inf
                  'latent_vec': None,
                  'latent_vec_dec': None,
                  'latent_type':'T5_vqvae',
                  'latent_num_layer': None,
                  'two_stage_vqvae': False,
                  'disentangled_vqvae': False,
                  'code_book_size': 10000,
                  'hard_code_select': False,
                  'pretrain_model_path':None,
                  'ema': True}

    data_dict = {'train_data_file': 'datasets/full/MathSymbolInf/both_tr.txt',
                 'test_data_file': 'datasets/full/MathSymbolInf/both_te.txt',
                 'overwrite_cache': True,
                 'inject_way': 'encoder_prefix',
                 'task': 'math_inf'} # recon: natural language, math_recon: symbolic language, math_inf

    training_dict = {'output_dir': 'checkpoints/t5vqvae_debug', 'overwrite_output_dir': True, 'do_train': True, 'do_eval': False,
                     'do_predict': False, 'evaluate_during_training': False, 'per_device_train_batch_size': 1,
                     'per_device_eval_batch_size': 1, 'per_gpu_train_batch_size': None, 'per_gpu_eval_batch_size': None,
                     'gradient_accumulation_steps': 1, 'learning_rate': 5e-05, 'weight_decay': 0.0, 'adam_epsilon': 1e-08,
                     'max_grad_norm': 1.0, 'num_train_epochs': 100, 'max_steps': -1, 'warmup_steps': 0,
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
    args.load_pretrain_t5 = False
    args.device = 'cpu'
    args.beta = 1.0
    args.log_interval = 10

    args.dvae_name = 'VRNN'
    args.config_file = 'cfg_vrnn.ini'

    set_seed(training_args.seed)
    dimension_dict = None
    srl_vocab = None

    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.INFO)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    fh = logging.FileHandler(os.path.join(args.output_dir, "train_log.log"), mode='w')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    args.logger = logger

    # ----------------------------------------- checking latent connection mode. --------------------------------------------
    if model_args.latent_type == 'T5_original':

        if model_args.t5_model_name == 't5-base':
            model = t5.load_t5_origin(model_args, training_args) if args.load_pretrain_t5 else t5.new_t5_origin(model_args, training_args)
        elif model_args.t5_model_name == "google/flan-t5-base":
            model = flan_t5.load_flan_t5_original(model_args, training_args) if args.load_pretrain_t5 else flan_t5.new_flan_t5_original(model_args, training_args)
        else:
            model = None
            exit('Error: wrong model name (two options: t5-base or google/flan-t5-base)')

    elif model_args.latent_type == 'T5_vae':
        model = t5.load_t5_vae(model_args, training_args) if args.load_pretrain_t5 else t5.new_t5_vae(model_args, training_args)

    elif model_args.latent_type == 'T5_ae':
        model = t5.load_t5_ae(model_args, training_args) if args.load_pretrain_t5 else t5.new_t5_ae(model_args, training_args)

    elif model_args.latent_type == 'T5_dvae':
        model_dvae, _ = build_model(args.dvae_name, config_file=args.config_file, device=args.device)
        model = t5.load_t5_dvae(model_args, training_args, model_dvae, load_model='T5') if args.load_pretrain_t5 else t5.new_t5_dvae(model_args, training_args, model_dvae)

    elif model_args.latent_type == 'T5_vqvae':
        # model_vqvae = VectorQuantizer(num_embeddings=512, embedding_dim=768, commitment_cost=0.25)
        dimension_dict = {"V": (0, 1000), "residual": (1000, args.code_book_size)}
        # TODO: create vocab.
        if args.disentangled_vqvae:
            srl_vocab = create_srl_vocab("datasets/full/explanations_vqvae.txt")
            print(srl_vocab)
            args.srl_embed_dim = len(srl_vocab)
            save_path = args.output_dir+"/srl_vocab.pickle"
            if not os.path.exists(save_path):
                os.makedirs(save_path)

        if args.ema:
            model_vqvae = VectorQuantizerEMA(num_embeddings=args.code_book_size, embedding_dim=model_args.ae_latent_size, commitment_cost=0.25, decay=0.99, disentanglement=args.disentangled_vqvae, loss="mse", **dimension_dict)
        else:
            model_vqvae = VectorQuantizer(num_embeddings=args.code_book_size, embedding_dim=model_args.ae_latent_size, commitment_cost=0.25, decay=0.99, disentanglement=args.disentangled_vqvae, loss="mse", **dimension_dict)

        model = t5.load_t5_vqvae(args, training_args, model_vqvae, load_model='T5-vqvae') if args.load_pretrain_t5 else t5.new_t5_vqvae(args, training_args, model_vqvae)

        if args.task in ['math_recon', 'math_inf']:
            # add special token [SEP]
            special_tokens_dict = {'sep_token': '[SEP]'}
            model.tokenizer.add_special_tokens(special_tokens_dict)
            model.tokenizer.add_tokens(['\\', '^', '{', '}'])

        model.t5_model.resize_token_embeddings(len(model.tokenizer))
    else:
        model = None
        exit('Error: wrong latent_type (three options: T5_original, T5_ae, T5_vae)')

    # ------------------------------------------------------------------------------------------------------------------------
    model.to(args.device)
    # Get datasets
    train_dataset, test_dataset = data.get_dataset(data_args, tokenizer=model.tokenizer, set_seq_size=model_args.set_seq_size, local_rank=training_args.local_rank, inject=args.inject_way, disentangle=dimension_dict, task=args.task, srl_vocab=srl_vocab)
    train_dataloader = data.get_dataloader(args, train_dataset)
    test_dataloader = data.get_dataloader(args, test_dataset)

    train(args, model, train_dataloader, test_dataloader, disentangled_vqvae=args.disentangled_vqvae)


if __name__ == '__main__':
    main()