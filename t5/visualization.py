import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import load_t5 as t5
from python_transformers import (MODEL_WITH_LM_HEAD_MAPPING, PreTrainedTokenizer, Trainer, set_seed)
import load_flan_t5 as flan_t5


def encoder_attention_visualizer(model):
    p1 = "blacktop is made of asphalt concrete"
    p2 = "asphalt has a smooth surface"
    type = "conjunction"
    t0 = [i for i in p1.split(" ") if i not in (" ", ',')]
    t1 = [i for i in p2.split(" ") if i not in (" ", ',')]
    text = 'inference type is '+type+' </s> '+' '.join(t0) + ' </s> ' + ' '.join(t1) + ' </s>'
    inp = model.tokenizer.batch_encode_plus([text], max_length=70, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
    input_ids = inp['input_ids']
    attention = model.get_attention(input_ids)
    input_token = [model.tokenizer.decode(i.item()) for i in input_ids[0] if i.item() > 0]

    # for h in range(12):
    h = 0
    attention_map = attention[1][11][0][h].detach().numpy()[:len(input_token), :len(input_token)]
    ax = sns.heatmap(attention_map, xticklabels=input_token, yticklabels=input_token)
    plt.savefig("figure_"+type+"_h_"+str(h))
    plt.show()


def decoder_cross_attention_visualizer(model):

    # construct input
    p1 = "blacktop is made of asphalt concrete"
    p2 = "asphalt has a smooth surface"
    c = "a blacktop has a smooth surface"
    type = "arg_substitution"
    t0 = [i for i in p1.split(" ") if i not in (" ", ',')]
    t1 = [i for i in p2.split(" ") if i not in (" ", ',')]
    c = [i for i in c.split(" ") if i not in (" ", ',')]
    # text = 'inference type is '+type+' </s> '+' '.join(t0) + ' </s> ' + ' '.join(t1) + ' </s>'
    text = ' '.join(t0) + ' </s> ' + ' '.join(t1) + ' </s>'
    inp = model.tokenizer.batch_encode_plus([text], max_length=20, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
    input_ids = inp['input_ids']

    text1 = '</s> ' + ' '.join(c)
    oup_ids = model.tokenizer.batch_encode_plus([text1], max_length=20, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
    decoder_input_ids = oup_ids["input_ids"]

    # feed into encoder.
    encoding = model.get_latent(input_ids)

    # feed into decoder and get the final cross-attention
    cross_attention = model.t5_model.decoder(input_ids=decoder_input_ids, encoder_hidden_states=encoding, output_attentions=True)[2][1]['CA0']
    print(cross_attention[0][11].shape)

    input_token = [model.tokenizer.decode(i.item()) for i in input_ids[0]] # if i.item() > 0
    output_token = [model.tokenizer.decode(i.item()) for i in decoder_input_ids[0]] # if i.item() > 0

    for h in range(12):
        attention_map = cross_attention[0][h].detach().numpy().T[:len(input_token), :len(output_token)] # q, k
        ax = sns.heatmap(attention_map, xticklabels=output_token, yticklabels=input_token)
        ax.set_title('L11 H'+str(h))
        # plt.savefig("figure_"+type+"_h_"+str(h))
        plt.show()


if __name__ == '__main__':
    model_dict = {'model_path': '', 't5_model_name': 't5-base', 'model_type': 't5', 'config_name': None,
                  'tokenizer_name': None, 'cache_dir': None, 'ae_latent_size': 1000, 'set_seq_size': 70,
                  'latent_vec': 'pooling', 'latent_type':'T5_original', 'latent_vec_dec': 'linear',
                  'pretrain_model_path':'checkpoints/checkpoints_without_inference_type/t5_base_original_loss_0.61/train'}

    data_dict = {'train_data_file': 'datasets/full/tr_data_with_fine_type.csv',
                 'test_data_file': 'datasets/full/te_data_with_fine_type.csv',
                 'overwrite_cache': True,
                 'inject_way': 'encoder_prefix'}

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
    # print(model)
    decoder_cross_attention_visualizer(model)
