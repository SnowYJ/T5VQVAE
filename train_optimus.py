import argparse
import time
import os
import json
from tqdm import tqdm

# from model import *
# from meter import AverageMeter
from optimus.utils import set_seed, logging, load_sample, conv_sent_dict, strip_eos, convert, millify, tqdm_joblib, load_sample_inf
from optimus.batchify import get_batches, get_batches_lm
# from vocab import Vocab
from optimus.data_access.definitions import ExplanationSemanticRoleCorpus, ExplanationInferenceCorpus
from optimus.examples.big_ae.utils import (weight_init, calc_iwnll, calc_rec, calc_mi, calc_au, BucketingDataLoader, TextDataset_Split, TextDataset_2Tokenizers, frange_cycle_linear, frange_cycle_zero_linear)
from optimus.examples.big_ae.modules import OptimusVAE
from python_transformers.optimization import AdamW, get_linear_schedule_with_warmup
from optimus.pytorch_transformers import (WEIGHTS_NAME, AdamW, WarmupLinearSchedule, BertConfig, BertForLatentConnector, BertTokenizer, GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer)
import torch
MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer),
    'bert': (BertConfig, BertForLatentConnector, BertTokenizer)
}
import logging
def add_args(parser):
    # Data Locations
    # 'entailmentbankREC' without conclusion for reconstruction only.
    # 'entailmentbankINF' sentences pair (premises, conclusion).
    # 'entailmentbankCON' with conclusion for strategy 2.
    parser.add_argument('--corpus', default='debug', metavar='C', required=False,
                        choices=['debug', 'wordnet', 'wiktionary', 'wikipedia',
                                 'entailmentbankREC', 'entailmentbankINF', 'entailmentbankCON'],
                        help='corpus to be used')
    parser.add_argument("--lm", type=str, required=False, nargs="+",
                        help="location of txt file with text for LM pre-training")
    parser.add_argument('--lm_ckpt', type=str, required=False,
                        help="location of pretrained LM")
    parser.add_argument('--input_train', type=str, required=False,
                        help="location of train vectors for Input conditioning")
    parser.add_argument('--input_eval', type=str, required=False,
                        help="location of eval vectors for Input conditioning")
    parser.add_argument('--save-dir', default='checkpoints', metavar='DIR',
                        help='directory to save checkpoints and outputs')
    parser.add_argument('--log-dir', metavar='DIR',
                        help='only used to copy log from localscratch')
    parser.add_argument("--w2v_weights", type=str, required=False,
                        help="path to pretrained embeddings to init")
    parser.add_argument('--local_rank', type=int, default=-1, metavar='N', help='DDP Local process rank.')


    # Data Settings
    parser.add_argument("--pretrain", action='store_true', help='pretrain LM flag')
    parser.add_argument('--load-model', default='', metavar='FILE', help='path to load checkpoint if specified')

    # Architecture arguments
    parser.add_argument('--dim_z', type=int, default=128, metavar='D', help='dimension of latent variable z')
    parser.add_argument('--dim_emb', type=int, default=512, metavar='D', help='dimension of word embedding')
    parser.add_argument('--dim_h', type=int, default=1024, metavar='D', help='dimension of hidden state per layer')
    parser.add_argument('--nlayers', type=int, default=1, metavar='N', help='number of layers')
    parser.add_argument('--dim_d', type=int, default=512, metavar='D', help='dim of hidden state in AAE discriminator')

    # Model arguments
    parser.add_argument('--pt_lm', default='t5-small', metavar='M',
                        choices=['t5-base', 't5-small', 'patrickvonplaten/t5-tiny-random'],
                        help='pre-trained emb LM')
    parser.add_argument('--model_type', default='dae', metavar='M',
                        choices=['beta', 'ann', 'dae', 'vae', 'aae', 'dm'],
                        help='which model to learn')
    parser.add_argument('--latent_spec', default='{"cont": 10,"disc": [20,2,2,3]}', type=json.loads)
    parser.add_argument('--cont_capacity', default="0.0,5.0,25000.0,30.0", type=str)
    parser.add_argument('--disc_capacity', default="0.0,5.0,25000.0,30.0", type=str)

    parser.add_argument('--eval_dis', action='store_true', help='evaluation')
    parser.add_argument('--print_traversal', action='store_true', help='print_traversal')
    parser.add_argument('--print_loss', action='store_true', help='print loss')
    parser.add_argument('--eval_interval', type=int, default=1, metavar='N', help='report eval')

    parser.add_argument('--lambda_kl', type=float, default=0, metavar='R',
                        help='weight for kl term in VAE')
    parser.add_argument('--lambda_adv', type=float, default=0, metavar='R',
                        help='weight for adversarial loss in AAE')
    parser.add_argument('--lambda_p', type=float, default=0, metavar='R',
                        help='weight for L1 penalty on posterior log-variance')
    parser.add_argument('--noise', default='0,0,0,0', metavar='P,P,P,K',
                        help='word drop prob, blank prob, substitute prob'
                             'max word shuffle distance')
    # Training arguments
    parser.add_argument('--dropout', type=float, default=0.5, metavar='DROP', help='dropout prob, 0 = no dropout')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR', help='learning rate')
    parser.add_argument('--decay_patience', type=float, default=0, help='decay patience')
    parser.add_argument('--decay_factor', type=float, default=0.1, help='decay factor')
    parser.add_argument('--epochs', type=int, default=30, metavar='N', help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N', help='batch size')

    # Others
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='report interval')
    parser.add_argument('--max-len', type=int, default=20, metavar='N',
                        help='max sequence length')
    parser.add_argument('--dec', default='greedy', metavar='M',
                        choices=['greedy', 'sample'], help='decoding algorithm')

    parser.add_argument('--exp', default='exp3', metavar='M',
                        choices=['exp1', 'exp2', 'exp3', 'exp3_2', 'exp4_train', 'exp4_gen',
                                 'exp_infer'],
                        help='type of batch to retrieve')

    # Optimus
    parser.add_argument('--latent_size', type=int, default=32, metavar='N', help='latent size of optimus')
    parser.add_argument('--model', default='optimus', metavar='M', choices=['optimus', 'conditional_optimus', 'TransformerCVAE'],
                        help='latent size of optimus')
    parser.add_argument('--latent_as_gpt_memory', default=True, help='optimus latent injection 1')
    parser.add_argument('--latent_as_gpt_emb', default=False, help='optimus latent injection 2')

    parser.add_argument('--use_pretrained_optimus', action='store_true', help='whether use pre-trained optimus')
    parser.add_argument('--dim_target_kl', default=1.0, type=float, help='optimus calculating kl')
    parser.add_argument('--fb_mode', default=0, type=int, help='optimus calculating kl')
    parser.add_argument('--beta', default=2.0, type=int, help='optimus final kl weight')
    parser.add_argument('--length_weighted_loss', action='store_true')
    parser.add_argument('--model_loss_func', default='beta_vae', choices=['beta_vae', 'tc_vae'], help='type of loss function (beta vae and tcvae)')
    parser.add_argument('--gradient_accumulation_steps', default=3, type=int)

    parser.add_argument('--inference_premises_com', action='store_true', help='NLI: p1 and p2 is feed into model as a single sentence')
    parser.add_argument('--inference_premises_sep', action='store_true', help='NLI: p1 and p2 are feed into model separately')


def load_optimus(args, logger):
    encoder_config_class, encoder_model_class, encoder_tokenizer_class = MODEL_CLASSES['bert']
    decoder_config_class, decoder_model_class, decoder_tokenizer_class = MODEL_CLASSES['gpt2']

    print('using optimus tokenizer!')
    tokenizer_encoder = encoder_tokenizer_class.from_pretrained('bert-base-cased', do_lower_case=False)
    tokenizer_decoder = decoder_tokenizer_class.from_pretrained('gpt2', do_lower_case=False)
    special_tokens_dict = {'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>'}
    num_added_toks = tokenizer_decoder.add_special_tokens(special_tokens_dict)
    print('We have added', num_added_toks, 'tokens to GPT2')
    logger.info('We have added %s tokens to GPT2', str(num_added_toks))
    pretrained_full_path = None

    if args.use_pretrained_optimus:
        path = args.pretrain_model_path
        pretrained_encoder_path = path + '/checkpoint-encoder'
        pretrained_decoder_path = path + '/checkpoint-decoder'
        pretrained_full_path = path +'/checkpoint-full/training.bin'

        model_encoder = encoder_model_class.from_pretrained(pretrained_encoder_path, latent_size=args.latent_size)
        model_encoder.to(args.device)
        model_decoder = decoder_model_class.from_pretrained(pretrained_decoder_path, latent_size=args.latent_size)
        model_decoder.to(args.device)
        model_decoder.resize_token_embeddings(len(tokenizer_decoder))
    else:
        encoder_config = encoder_config_class.from_pretrained('bert-base-cased')
        model_encoder = encoder_model_class.from_pretrained('bert-base-cased', cache_dir=None, config=encoder_config, latent_size=args.latent_size) # ,from_tf=bool('.ckpt' in 'bert-base-cased')

        decoder_config = decoder_config_class.from_pretrained('gpt2')
        model_decoder = decoder_model_class.from_pretrained('gpt2', cache_dir=None, config=decoder_config, latent_size=args.latent_size, latent_as_gpt_emb=False, latent_as_gpt_memory=True) # from_tf=bool('.ckpt' in 'gpt2')

        model_decoder.resize_token_embeddings(len(tokenizer_decoder))
        model_encoder.resize_token_embeddings(len(tokenizer_encoder))

        print('Optimus: pretrained BERT & GPT2 are successfully loaded')
        logger.info('Optimus: pretrained BERT & GPT2 are successfully loaded')

    model = OptimusVAE(model_encoder, model_decoder, tokenizer_encoder, tokenizer_decoder, args)

    if args.use_pretrained_optimus:
        checkpoint = torch.load(pretrained_full_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print('Optimus: pretrained optimus is successfully loaded')

    return tokenizer_encoder, tokenizer_decoder, model


def get_optimizers(args, model, num_training_steps):
    """
        Setup the optimizer and the learning rate scheduler, modified for when training with a VAE with an input-decoder.
    """
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-08)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    return optimizer, scheduler


def evaluate(model, batches, role_pad=None, data_len=0, args=None, logger=None):
    model.eval()
    final_loss = 0
    with torch.no_grad():
        for i in range(len(batches)):
            inputs, labels = batches[i][0].T, batches[i][1].T
            input_roles, label_roles = ((batches[i][2].T).long(), (batches[i][3].T).long()) if args.exp not in ['exp1', 'exp4_train', 'exp4_gen', 'exp_infer'] else (None, None)
            inputs_1 = batches[i][2].T if args.inference_premises_sep else None
            loss_rec, loss_kl, loss = model.autoenc(inputs.long(), labels.long(), input_roles, label_roles, args, role_pad, data_len, infer=True if args.exp == 'exp_infer' else False, sep_infer_input=inputs_1)

            if (i + 1) % args.log_interval == 0:
                print('| epoch {:3d} | {:5d}/{:5d} batches | test loss {:.4f} , kl {:.4f} , latent rec {:.4f}'.format(1, i, len(batches), loss, loss_kl, loss_rec))
                logger.info('| epoch {:3d} | {:5d}/{:5d} batches | test loss {:.4f} , kl {:.4f} , latent rec {:.4f}'.format(1, i, len(batches), loss, loss_kl, loss_rec))

            final_loss += loss

    return final_loss/len(batches)


def train_func(args, model, train_batches=None, test_batches=None, logger=None):
    n_iter = int(args.epochs) * len(train_batches)
    gradient_accumulation_steps = args.gradient_accumulation_steps
    beta_t_list = frange_cycle_zero_linear(n_iter, start=0.0, stop=args.beta,  n_cycle=5, ratio_increase=0.25, ratio_zero=0.25)
    len_train_dataloader = len(train_batches)
    t_total = int(len_train_dataloader // args.gradient_accumulation_steps * args.epochs)
    optimizer, scheduler = get_optimizers(args, model=model, num_training_steps=t_total)
    min_loss = 100000
    model.zero_grad()

    print("**** start running checking training loss in train_log file ****")
    logger.info("***** Running training *****")
    logger.info("  Task = %s", args.exp)
    logger.info("  Latent Size = %s", args.latent_size)
    logger.info("  Encoder = %s", 'BERT')
    logger.info("  Decoder = %s", 'GPT2')
    logger.info("  Num Epochs = %s", args.epochs)
    logger.info("  KL annealing = start from %s to %s", 0, args.beta)
    logger.info("  batch size per device = %s", args.batch_size)

    for epoch in range(args.epochs):
        model.train()
        print('----'*30)
        logger.info('----'*30)
        # indices = range(len(train_batches))
        for i, idx in enumerate(range(len(train_batches))):

            cur_beta = beta_t_list[i + epoch*len_train_dataloader]
            model.args.beta = cur_beta
            model.args.lamb = 0
            model.args.fb_mode = 0 if cur_beta == 0.0 else 1

            # randomly choose a number > 2 will make optimus to autoencoder.
            # model.args.fb_mode = 3

            inputs, labels = train_batches[idx][0].T, train_batches[idx][1].T
            input_roles, label_roles = ((train_batches[idx][2].T).long(), train_batches[idx][3].T.long()) if args.exp not in ['exp1', 'exp4_train', 'exp4_gen', 'exp_infer'] else (None, None)

            # only used in inference task with separate premises.
            inputs_1 = train_batches[idx][2].T if args.inference_premises_sep else None

            loss_rec, loss_kl, loss = model.autoenc(inputs.long(), labels.long(), input_roles, label_roles, args, None, infer=True if args.exp == 'exp_infer' else False, sep_infer_input=inputs_1)

            if gradient_accumulation_steps > 1:
                loss /= gradient_accumulation_steps
                loss_kl /= gradient_accumulation_steps
                loss_rec /= gradient_accumulation_steps

            loss.backward()

            if (i + 1) % args.log_interval == 0:
                print('| epoch {:3d} | {:5d}/{:5d} batches | train loss {:.4f} , kl {:.4f} , latent rec {:.4f}'.format(epoch + 1, i, len_train_dataloader, loss, loss_kl, loss_rec))
                logger.info('| epoch {:3d} | {:5d}/{:5d} batches | train loss {:.4f} , kl {:.4f} , latent rec {:.4f}'.format(epoch + 1, i, len_train_dataloader, loss, loss_kl, loss_rec))

            if (i + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # max_grad_norm: 1.0
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()

        test_loss = evaluate(model, test_batches, args.model, args=args, logger=logger)

        if min_loss > test_loss:
            min_loss = test_loss
            print('| epoch {:3d} end | avg test loss {:.4f}| saving model!'.format(epoch + 1, test_loss))
            logger.info('| epoch {:3d} end | avg test loss {:.4f}| saving model!'.format(epoch + 1, test_loss))
            model_encoder_to_save = model.module.encoder if hasattr(model, 'module') else model.encoder
            model_decoder_to_save = model.module.decoder if hasattr(model, 'module') else model.decoder
            model_to_save = model.module if hasattr(model, 'module') else model

            # save encoder & decoder & full
            output_encoder_dir = os.path.join(args.save_dir, 'checkpoint-encoder')
            output_decoder_dir = os.path.join(args.save_dir, 'checkpoint-decoder')
            output_full_dir = os.path.join(args.save_dir, 'checkpoint-full')

            if not os.path.exists(output_encoder_dir):
                os.makedirs(output_encoder_dir)
            if not os.path.exists(output_decoder_dir):
                os.makedirs(output_decoder_dir)
            if not os.path.exists(output_full_dir):
                os.makedirs(output_full_dir)

            model_encoder_to_save.save_pretrained(output_encoder_dir)
            model_decoder_to_save.save_pretrained(output_decoder_dir)

            checkpoint = {
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args
            }
            torch.save(checkpoint, os.path.join(output_full_dir, 'training.bin'))
        else:
            print('| epoch {:3d} end | avg test loss {:.4f}| not saving model!'.format(epoch + 1, test_loss))
            logger.info('| epoch {:3d} end | avg test loss {:.4f}| not saving model!'.format(epoch + 1, test_loss))


def main(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)

    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(args.save_dir, "train_log.log"), mode='w')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    tokenizer_encoder, tokenizer_decoder, model = load_optimus(args, logger)
    model = model.to(device)

    # ------------------------------------------------------------------------------------------------------------------------------
    corpus = ExplanationSemanticRoleCorpus(args.corpus) if args.exp == 'exp1' else ExplanationInferenceCorpus(args.corpus)
    infer_type = False if args.inference_premises_sep else True
    train, valid, test = load_sample(corpus) if args.exp == 'exp1' else load_sample_inf(corpus)
    conv_sent_dict_param = {'emb_tokenizer': tokenizer_encoder, 'decode_tokenizer': None if args.model == 'rnn' else tokenizer_decoder, 'model': args.model, 'args': args}

    train_sents, valid_sents = [], []
    for sent in tqdm(train):
        tr_temp = conv_sent_dict(sent, **conv_sent_dict_param)
        train_sents.append(tr_temp)

    for sent in tqdm(valid):
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
    train_func(args, model, train_batches=train_batches, test_batches=valid_batches, logger=logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    dic = {
        'batch_size':64,
        'cont_capacity':'0.0,5.0,25000.0,30.0',
        'corpus':'entailmentbankVQVAE',
        'dec':'greedy',
        'decay_factor':0.1,
        'decay_patience':0,
        'dim_d':512,
        'dim_emb':512,
        'dim_h':1024,
        'dim_z':256,
        'latent_size': 32,
        'model_loss_func': 'beta_vae',
        'gradient_accumulation_steps': 1,
        'learning_rate': 5e-05,
        'device': 'cuda',
        'model': 'optimus',
        'latent_as_gpt_memory': True,
        'latent_as_gpt_emb': False,
        'use_pretrained_optimus': False,
        'pretrain_model_path': './checkpoint-32',
        'inference_premises_com': True,
        'inference_premises_sep': False,
        'dim_target_kl': 1.0,
        'fb_mode': 0,
        'length_weighted_loss': False,
        'beta': 1.0,
        'disc_capacity':'0.0,5.0,25000.0,30.0',
        'dropout': 0.5,
        'epochs': 30,
        'eval_dis': False,
        'eval_interval':1,
        'exp':'exp1',
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
        'log_interval':10,
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
        'save_dir': 'checkpoints/test',
        'seed':0,
        'w2v_weights':None
    }
    args = argparse.Namespace(**dic)
    main(args)

