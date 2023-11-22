from python_transformers import PreTrainedModel
from torch.nn import CrossEntropyLoss
from python_transformers import T5ForConditionalGeneration, T5Tokenizer
from t5.load_vae import _get_vae
from t5.load_ae import _get_ae
from python_transformers import AutoConfig, CONFIG_MAPPING
import torch
from torch import nn
from sentence_transformers import SentenceTransformer
from dvae.utils import myconf, get_logger, loss_ISD, loss_KLD, loss_MPJPE, loss_MSE
import torch.nn.functional as F
from torch import einsum

# ================================================= Load T5 ============================================================
def _get_config(model_args):
    if model_args.config_name:
        return AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_path:
        return AutoConfig.from_pretrained(model_args.model_path, cache_dir=model_args.cache_dir)
    else:
        # print("You are instantiating a new config instance from scratch.")
        # print(CONFIG_MAPPING[model_args.model_type]())
        return CONFIG_MAPPING[model_args.model_type]()


def _get_t5_vae_requirements(model_args, training_args):
    config = _get_config(model_args)
    name = model_args.t5_model_name  # "t5-base"
    tokenizer = T5Tokenizer.from_pretrained(name)
    t5_model = T5ForConditionalGeneration.from_pretrained(name, latent_size=None)
    # t5_model, tokenizer = _get_t5_model(model_args.t5_model_name, model_args.tokenizer_name, model_args.cache_dir)
    vae = _get_vae(t5_model.config, model_args, training_args)
    return config, t5_model, tokenizer, vae


def _get_t5_ae_requirements(model_args, training_args):
    config = _get_config(model_args)
    name = model_args.t5_model_name  # "t5-base"
    tokenizer = T5Tokenizer.from_pretrained(name)

    t5_model = T5ForConditionalGeneration.from_pretrained(name, latent_size=None)
    # t5_model, tokenizer = _get_t5_model(model_args.t5_model_name, model_args.tokenizer_name, model_args.cache_dir)
    ae = _get_ae(t5_model.config, model_args, training_args)
    return config, t5_model, tokenizer, ae


def _get_t5_origin_requirements(model_args, training_args):
    config = _get_config(model_args)
    name = model_args.t5_model_name  # "t5-base"
    tokenizer = T5Tokenizer.from_pretrained(name)
    t5_model = T5ForConditionalGeneration.from_pretrained(name)
    return config, t5_model, tokenizer


# =======================================================================================================================

def new_t5_vae(model_args, training_args):
    config, t5_model, tokenizer, vae = _get_t5_vae_requirements(model_args, training_args)
    return t5_AE(config, t5_model, vae, model_args.set_seq_size, tokenizer, model_args, training_args)


def load_t5_vae(model_args, training_args):
    config, t5_model, tokenizer, vae = _get_t5_vae_requirements(model_args, training_args)
    t5_ae = t5_AE(config, t5_model, vae, model_args.set_seq_size, tokenizer, model_args, training_args)
    checkpoint = torch.load(model_args.pretrain_model_path, map_location=torch.device('cpu'))
    t5_ae.load_state_dict(checkpoint)
    print('loading pretrained t5_vae successful.')
    return t5_ae


def load_t5_ae(model_args, training_args):
    config, t5_model, tokenizer, vae = _get_t5_ae_requirements(model_args, training_args)
    t5_ae = t5_AE(config, t5_model, vae, model_args.set_seq_size, tokenizer, model_args, training_args)

    # if model_args.latent_vec in ['sentenceT5', 'sentenceT5_as_mem', 'sentenceT5_as_input', 'sentenceT5_as_output']:
    #     from load_sentenceT5 import load_sentenceT5_weight
    #     from sentence_transformers import SentenceTransformer
    #     sent_t5 = SentenceTransformer('sentence-transformers/sentence-t5-base')
    #     new_state_dict = load_sentenceT5_weight(t5_ae, sent_t5)
    #     t5_ae.load_state_dict(new_state_dict)
    #
    #     s1 = (t5_ae.state_dict()['t5_model.lm_head.weight'] == t5_ae.state_dict()['t5_model.shared.weight']).all()
    #     s2 = (t5_ae.state_dict()['t5_model.encoder.embed_tokens.weight'] == t5_ae.state_dict()['t5_model.shared.weight']).all()
    #     s3 = (t5_ae.state_dict()['t5_model.decoder.embed_tokens.weight'] == t5_ae.state_dict()['t5_model.shared.weight']).all()
    #
    #     if s1 and s2 and s3:
    #         print('loading sentenceT5 to T5encoder successful.')
    #     else:
    #         exit('ERROR in new_t5_ae')

    checkpoint = torch.load(model_args.pretrain_model_path, map_location=torch.device('cpu'))
    t5_ae.load_state_dict(checkpoint, strict=False)
    print('loading pretrained t5_ae successful.')
    return t5_ae


def new_t5_ae(model_args, training_args):
    config, t5_model, tokenizer, vae = _get_t5_ae_requirements(model_args, training_args)
    t5_ae = t5_AE(config, t5_model, vae, model_args.set_seq_size, tokenizer, model_args, training_args)

    if model_args.latent_vec in ['sentenceT5', 'sentenceT5_as_mem', 'sentenceT5_as_input', 'sentenceT5_as_output']:
        from load_sentenceT5 import load_sentenceT5_weight
        from sentence_transformers import SentenceTransformer
        sent_t5 = SentenceTransformer('sentence-transformers/sentence-t5-base')
        new_state_dict = load_sentenceT5_weight(t5_ae, sent_t5)
        t5_ae.load_state_dict(new_state_dict)

        s1 = (t5_ae.state_dict()['t5_model.lm_head.weight'] == t5_ae.state_dict()['t5_model.shared.weight']).all()
        s2 = (t5_ae.state_dict()['t5_model.encoder.embed_tokens.weight'] == t5_ae.state_dict()[
            't5_model.shared.weight']).all()
        s3 = (t5_ae.state_dict()['t5_model.decoder.embed_tokens.weight'] == t5_ae.state_dict()[
            't5_model.shared.weight']).all()

        if s1 and s2 and s3:
            print('loading sentenceT5 to T5encoder successful.')
        else:
            exit('ERROR in new_t5_ae')

    return t5_ae


def load_t5_origin(model_args, training_args):
    config, t5_model, tokenizer = _get_t5_origin_requirements(model_args, training_args)
    t5 = t5_AE(config, t5_model, None, model_args.set_seq_size, tokenizer, model_args, training_args)
    checkpoint = torch.load(model_args.pretrain_model_path, map_location=torch.device('cpu'))
    t5.load_state_dict(checkpoint)
    print('loading pretrained t5 successful.')
    return t5


def new_t5_origin(model_args, training_args):
    config, t5_model, tokenizer = _get_t5_origin_requirements(model_args, training_args)
    return t5_AE(config, t5_model, None, model_args.set_seq_size, tokenizer, model_args, training_args)


def load_t5_dvae(model_args, training_args, dvae, load_model=None):
    assert model_args.latent_type == 'T5_dvae'

    config, t5_model, tokenizer = _get_t5_origin_requirements(model_args, training_args)
    t5 = t5_AE(config, t5_model, dvae, model_args.set_seq_size, tokenizer, model_args, training_args)
    if load_model == 'T5':
        checkpoint = torch.load(model_args.pretrain_model_path, map_location=torch.device('cpu'))
        t5_model.load_state_dict(checkpoint)
        print('loading pretrained t5 successful.')
    elif load_model == 'dvae':
        checkpoint = torch.load(model_args.pretrain_model_path, map_location=torch.device('cpu'))
        dvae.load_state_dict(checkpoint)
        print('loading pretrained DVAE successful.')
    elif load_model == 'T5-dvae':
        checkpoint = torch.load(model_args.pretrain_model_path, map_location=torch.device('cpu'))
        t5.load_state_dict(checkpoint)
        print('loading pretrained t5-DVAE successful.')
    else:
        exit()

    return t5


def new_t5_dvae(model_args, training_args, dvae):
    assert model_args.latent_type == 'T5_dvae'
    config, t5_model, tokenizer = _get_t5_origin_requirements(model_args, training_args)
    t5 = t5_AE(config, t5_model, dvae, model_args.set_seq_size, tokenizer, model_args, training_args)
    return t5


def new_t5_vqvae(model_args, training_args, vqvae):
    assert model_args.latent_type == 'T5_vqvae'
    # assert model_args.use_srl_embedding == True and dimension_dict is not None

    config, t5_model, tokenizer = _get_t5_origin_requirements(model_args, training_args)
    t5 = t5_AE(config, t5_model, vqvae, model_args.set_seq_size, tokenizer, model_args, training_args)
    return t5


def load_t5_vqvae(model_args, training_args, vqvae, load_model=None):
    assert model_args.latent_type == 'T5_vqvae'
    # assert model_args.use_srl_embedding == True and dimension_dict is not None

    config, t5_model, tokenizer = _get_t5_origin_requirements(model_args, training_args)

    if model_args.task in ['math_recon', 'math_inf']:
        # add special token [SEP]
        special_tokens_dict = {'sep_token': '[SEP]'}
        tokenizer.add_special_tokens(special_tokens_dict)
        tokenizer.add_tokens(['\\', '^', '{', '}'])
        t5_model.resize_token_embeddings(len(tokenizer))

    t5 = t5_AE(config, t5_model, vqvae, model_args.set_seq_size, tokenizer, model_args, training_args)

    if load_model == 'T5':
        checkpoint = torch.load(model_args.pretrain_model_path, map_location=torch.device('cpu'))
        t5_model.load_state_dict(checkpoint)
        print('loading pretrained t5 successful.')

    elif load_model == 'vqvae':
        checkpoint = torch.load(model_args.pretrain_model_path, map_location=torch.device('cpu'))
        vqvae.load_state_dict(checkpoint)
        print('loading pretrained DVAE successful.')

    elif load_model == 'T5-vqvae':
        # # add special token [SEP]
        # special_tokens_dict = {'sep_token': '[SEP]'}
        # tokenizer.add_special_tokens(special_tokens_dict)
        # t5_model.resize_token_embeddings(len(tokenizer))

        checkpoint = torch.load(model_args.pretrain_model_path, map_location=torch.device('cpu'))
        t5.load_state_dict(checkpoint, strict=False)
        print('loading pretrained t5-VQVAE successful.')

    else:
        exit()

    return t5


# ================================================== T5 + (V, D, VQ, .) AE =============================================

class t5_AE(PreTrainedModel):
    # base_model_prefix = 't5_vae'
    def __init__(self, config, t5_model, vae, set_seq_size, tokenizer, model_args, training_args):
        super().__init__(config=config)
        self.t5_model = t5_model

        if model_args.latent_type == 'T5_vae':
            self.vae = vae

        elif model_args.latent_type == 'T5_ae':
            self.ae = vae

        elif model_args.latent_type == 'T5_dvae':
            self.dvae = vae

        elif model_args.latent_type == 'T5_vqvae':
            self.vqvae = vae

            if model_args.task == 'inference_sep' and model_args.two_stage_vqvae == False:
                # new T5 encoder as inference model, it can be any kind of model.
                from transformers import T5EncoderModel
                T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
                self.inference_model = T5EncoderModel.from_pretrained("t5-base")
                # self.inference_model_embed = nn.Linear(model_args.ae_latent_size, model_args.set_seq_size*2)

            if model_args.disentangled_vqvae:
                # define srl input embedding layer to learn residual network under T5 encoder.
                self.t5_model.encoder.srl_embed = nn.Embedding(model_args.srl_embed_dim, t5_model.config.d_model)
                self.t5_model.encoder.srl_embed.weight.data.normal_()
                # self.bilinear = nn.Bilinear(t5_model.config.d_model, t5_model.config.d_model, t5_model.config.d_model)

            if model_args.hard_code_select:
                # self.logit_transform = nn.Linear(t5_model.config.d_model, model_args.code_book_size)
                hidden_size = 400
                linear_transform = []
                linear_transform.append(nn.Sequential(nn.Linear(t5_model.config.d_model*set_seq_size, set_seq_size*hidden_size), nn.Dropout(p=0.4), nn.ReLU()))
                linear_transform.append(nn.Sequential(nn.Linear(set_seq_size*hidden_size, set_seq_size*hidden_size), nn.Dropout(p=0.4), nn.ReLU()))
                linear_transform.append(nn.Sequential(nn.Linear(set_seq_size*hidden_size, set_seq_size*hidden_size), nn.Dropout(p=0.4), nn.ReLU()))
                linear_transform.append(nn.Sequential(nn.Linear(set_seq_size*hidden_size, set_seq_size*hidden_size), nn.Dropout(p=0.4), nn.ReLU()))
                linear_transform.append(nn.Linear(set_seq_size*hidden_size, set_seq_size*hidden_size))
                self.linear_transform = nn.Sequential(*linear_transform)
                self.logit_transform = nn.Linear(hidden_size, model_args.code_book_size)

                # self.enc_context_linear = nn.Linear(t5_model.config.d_model, t5_model.config.d_model)
                # self.dec_context_linear = nn.Linear(t5_model.config.d_model, t5_model.config.d_model)
                # self.conv = nn.Conv2d(in_channels=set_seq_size, out_channels=set_seq_size * t5_model.config.d_model, kernel_size=1, groups=set_seq_size, bias=True)

            # self.t5_model.encoder.srl_embed = nn.Embedding(model_args.srl_embed_dim, t5_model.config.d_model)
            # self.t5_model.encoder.srl_embed.weight.data.normal_()
        else:
            assert model_args.latent_type == 'T5_original'

        if model_args.latent_vec in ['sentenceT5', 'sentenceT5_as_mem', 'sentenceT5_as_input', 'sentenceT5_as_output']:
            size1, size2 = t5_model.state_dict()['shared.weight'].shape
            self.enc_embed_weight = nn.Embedding(size1, size2)

        self.config = config
        self.set_seq_size = set_seq_size
        self.tokenizer = tokenizer
        self.latent_type = model_args.latent_type
        self.latent_vec = model_args.latent_vec
        self.model_args = model_args

        self.batch_size = training_args.per_device_train_batch_size
        self.set_seq_size = model_args.set_seq_size
        self.d_model = t5_model.config.d_model
        self.codebook_size = model_args.code_book_size

    def _decoder_logits(self, decoder_input_ids, encoding, encoder_attention_mask):
        # decoder_attention_mask = (decoder_input_ids>0).long
        if self.latent_vec in ['pooling_as_mem', 'attention_as_mem', 'shrinking_as_mem', 'sentenceT5_as_mem']:
            batch_size = encoding.shape[0]
            sequence_size = encoding.shape[1]
            past_key_value_states = encoding.view(batch_size, 12, sequence_size, -1)
            sequence_output = self.t5_model.decoder(input_ids=decoder_input_ids, encoder_hidden_states=encoding,
                                                    latent_mem=past_key_value_states)[0]

        elif self.latent_vec in ['pooling_as_input', 'attention_as_input', 'shrinking_as_input', 'sentenceT5_as_input']:
            sequence_output = \
            self.t5_model.decoder(input_ids=decoder_input_ids, encoder_hidden_states=None, latent_mem=None,
                                  latent_input=encoding)[0]

        elif self.latent_vec in ['attention_as_output', 'pooling_as_output', 'shrinking_as_output',
                                 'sentenceT5_as_output']:
            sequence_output = \
            self.t5_model.decoder(input_ids=decoder_input_ids, encoder_hidden_states=None, latent_mem=None,
                                  latent_input=None)[0]
            sequence_output += encoding
        else:
            sequence_output = self.t5_model.decoder(input_ids=decoder_input_ids, encoder_hidden_states=encoding,
                                                    output_attentions=True)
            sequence_output = sequence_output[0]

        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (self.t5_model.model_dim ** -0.5)
        logits = self.t5_model.lm_head(sequence_output)
        return logits

    def decoder_loss(self, outputs, labels, encoding, ignore_index=0, encoder_attention_mask=None):
        # shift right to make it started with 1
        # decoder_input_ids = self.t5_model._shift_right(labels)
        """
        the decoder input looks like: [1, w1, w2, ..., wn, 0, 0, ...]
        the decoder label looks like: [w1, w2, ..., wn, 1, 0, 0, ...]
        """
        decoder_input_ids = outputs.contiguous()
        labels = labels.contiguous()

        logits = self._decoder_logits(decoder_input_ids, encoding, encoder_attention_mask)
        loss_fct = CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.long().view(-1))
        # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
        return loss

    def get_latent(self, input_ids):
        attention_mask = input_ids.ne(self.config.pad_token_id).long()
        if self.latent_vec in ['sentenceT5', 'sentenceT5_as_mem', 'sentenceT5_as_input', 'sentenceT5_as_output']:
            input_embed = self.enc_embed_weight(input_ids)
            encoding = self.t5_model.encoder(input_ids=None, inputs_embeds=input_embed, attention_mask=attention_mask)[
                0]
        else:
            encoding = self.t5_model.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]

        if self.latent_type == 'T5_vae':
            return self.vae(encoding, attention_mask=attention_mask, just_get_latent=True)
        elif self.latent_type == 'T5_ae':
            return self.ae(encoding, attention_mask=attention_mask, just_get_latent=True)
        else:
            return encoding

    def get_hidden(self, input_ids):
        attention_mask = input_ids.ne(self.config.pad_token_id).long()

        if self.latent_vec in ['sentenceT5', 'sentenceT5_as_mem', 'sentenceT5_as_input', 'sentenceT5_as_output']:
            input_embed = self.enc_embed_weight(input_ids)
            encoding = self.t5_model.encoder(input_ids=None, inputs_embeds=input_embed, attention_mask=attention_mask)[
                0]
        else:
            encoding = self.t5_model.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]

        if self.latent_type == 'T5_vae':
            return self.vae(encoding, attention_mask=attention_mask, just_get_encoding=True)
        elif self.latent_type == 'T5_ae':
            return self.ae(encoding, attention_mask=attention_mask, just_get_encoding=True)
        else:
            return encoding

    def get_attention(self, input_ids):
        attention_mask = input_ids.ne(self.config.pad_token_id).long()
        attention = self.t5_model.encoder(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        return attention

    def get_cross_attention(self, input_ids, output_ids):
        pass

    def forward(self, input_ids, output_ids, label_ids, srl_ids=None, input1_ids=None, latent_num_layer=None,
                two_stage_vqvae=False, alpha=1.0, srl_embed_ids=None):
        """
        input_ids: input of encoder
        output_ids: input of decoder
        label_ids: output of decoder
        srl_ids: semantic role label, it is only used for disentangled sentence space.
        input1_ids: the second input of encoder, which is only used for "separate premises => conclusion" inference task.
        latent_num_layer: 0-N (T5base is 12). The first N layer will be considered as encoder.
        two_stage_vqvae: True or False. If true, the T5 will be trained separately. First train AutoEncoder (0:N layer). Then, train inference model (N+1:12 layer).
        """
        recon_loss, reg_loss, loss_KL = 0, 0, 0
        attention_mask = input_ids.ne(self.config.pad_token_id).long()

        if self.latent_vec in ['sentenceT5', 'sentenceT5_as_mem', 'sentenceT5_as_input', 'sentenceT5_as_output']:
            assert self.model_args.latent_type == 'T5_ae'
            input_embed = self.enc_embed_weight(input_ids)
            encoding = self.t5_model.encoder(input_ids=None, inputs_embeds=input_embed, attention_mask=attention_mask)[0]
        else:
            if input1_ids is None:
                # Task: (1) inference task with combined premises and (2) autoencoding task.
                if self.latent_type is not 'T5_vqvae':
                    encoding = self.t5_model.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
                else:
                    # if latent_num_layer is None. VQVAE is work between encoder and decoder. Otherwise, it work in the middle of Encoder.
                    enc_out = self.t5_model.encoder(input_ids=input_ids, attention_mask=attention_mask, srl=srl_ids,
                                                    latent_num_layer=latent_num_layer, vqvae=self.vqvae,
                                                    two_stage_vqvae=two_stage_vqvae, srl_embed_ids=srl_embed_ids,
                                                    residual_alpha=0.7)
                    recon_loss, encoding, perplexity = enc_out[-1], enc_out[0], None
            else:
                assert self.model_args.latent_type == 'T5_vqvae'
                # Task: inference task with separated premises.
                if latent_num_layer is None:
                    # Whole Encoder
                    # encoding0 = self.t5_model.encoder(input_ids=input_ids, attention_mask=input_ids.ne(self.config.pad_token_id).long())[0]
                    # encoding1 = self.t5_model.encoder(input_ids=input1_ids, attention_mask=input1_ids.ne(self.config.pad_token_id).long())[0]
                    # encoding = torch.cat((encoding0, encoding1), 1)
                    encoding0 = self.t5_model.encoder(input_ids=input_ids,
                                                      attention_mask=input_ids.ne(self.config.pad_token_id).long(),
                                                      latent_num_layer=latent_num_layer, vqvae=self.vqvae)[0]
                    encoding1 = self.t5_model.encoder(input_ids=input1_ids,
                                                      attention_mask=input1_ids.ne(self.config.pad_token_id).long(),
                                                      latent_num_layer=latent_num_layer, vqvae=self.vqvae)[0]

                    tmp_output_ids = torch.concat((output_ids[:, 1:],
                                                   torch.zeros(output_ids.shape[0], 1, device=output_ids.device,
                                                               dtype=torch.long)), 1)
                    encoding2 = self.t5_model.encoder(input_ids=tmp_output_ids,
                                                      attention_mask=tmp_output_ids.ne(self.config.pad_token_id).long(),
                                                      latent_num_layer=latent_num_layer, vqvae=self.vqvae)[0]

                    encoding0 = self.vqvae(encoding0)[1]  # p1
                    encoding1 = self.vqvae(encoding1)[1]  # p2
                    encoding2 = self.vqvae(encoding2)[1]  # c

                    encoding01 = torch.cat((encoding0, encoding1), 1)
                    attention_mask = torch.cat(
                        (input_ids.ne(self.config.pad_token_id).long(), input1_ids.ne(self.config.pad_token_id).long()),
                        1)
                    # TODO: feed into an inference model - new T5 encoder.
                    encoding = self.inference_model.encoder(inputs_embeds=encoding01, attention_mask=attention_mask)[0]
                    # encoding = torch.split(encoding, input_ids.shape[1], dim=1)[0].contiguous()
                    # print(encoding.shape)
                    # encoding = self.inference_model_embed(encoding)
                    # print(encoding.shape)
                    # exit()
                    # TODO: calculate the mse between golden conclusion encoding and inferred conclusion encoding.
                    # encoding, encoding2
                    # recon_loss = F.mse_loss(encoding, encoding2.detach(), reduction='mean') # /torch.tensor(encoding.shape[1], device=encoding.device)

                else:
                    # Separate Encoder
                    if two_stage_vqvae:
                        encoding0 = self.t5_model.encoder(input_ids=input_ids,
                                                          attention_mask=input_ids.ne(self.config.pad_token_id).long(),
                                                          latent_num_layer=latent_num_layer, vqvae=self.vqvae)[0]
                        encoding1 = self.t5_model.encoder(input_ids=input1_ids,
                                                          attention_mask=input1_ids.ne(self.config.pad_token_id).long(),
                                                          latent_num_layer=latent_num_layer, vqvae=self.vqvae)[0]
                        encoding = torch.cat((encoding0, encoding1), 1)
                        attention_mask = torch.cat((input_ids.ne(self.config.pad_token_id).long(),
                                                    input1_ids.ne(self.config.pad_token_id).long()), 1)
                        encoding = self.t5_model.encoder(input_ids=encoding, attention_mask=attention_mask,
                                                         latent_num_layer=latent_num_layer, vqvae=self.vqvae,
                                                         is_inf=True)[0]
                    else:
                        encoding = None
                        exit("Error: for separating inference task.")

        # vae or ae.
        if self.latent_type == 'T5_vae':
            recon_loss, reg_loss, encoding, loss_KL = self.vae(encoding, attention_mask)

        elif self.latent_type == 'T5_ae':
            recon_loss, reg_loss, encoding = self.ae(encoding, attention_mask)

        elif self.latent_type == 'T5_dvae':
            encoding1 = self.dvae(encoding)
            recon_loss = loss_MSE(encoding, encoding1)
            loss_KL = loss_KLD(self.dvae.z_mean, self.dvae.z_logvar, self.dvae.z_mean_p, self.dvae.z_logvar_p)
            encoding = encoding1

        elif self.latent_type == 'T5_vqvae' and latent_num_layer is None and input1_ids is None:
            # it will only be used at the end of encoder.

            if self.model_args.hard_code_select:
                # using gumbel softmax to select codebook embedding.
                input_encoding = encoding * attention_mask.unsqueeze(-1)
                enc_embedding = torch.flatten(input_encoding, start_dim=1)
                compress_enc = self.linear_transform(enc_embedding).reshape(encoding.shape[0], self.set_seq_size, -1)
                logits = self.logit_transform(compress_enc)
                one_hot = F.gumbel_softmax(logits, tau=0.9, dim=1, hard=True) # False: reparametrization trick True: Straight-through trick
                indices_tensor = torch.argmax(one_hot, dim=-1)
                encoding = self.vqvae._embedding(indices_tensor)
                recon_loss = F.mse_loss(encoding*attention_mask.unsqueeze(-1), input_encoding)
                exit("not work")

                # # using gumbel softmax to select codebook embedding.
                # enc_embedding = encoding
                # # TODO: performing mask and add padding
                # logits = self.logit_transform(encoding)
                # one_hot = F.gumbel_softmax(logits, tau=0.9, dim=1, hard=True) # False: reparametrization trick True: Straight-through trick
                # indices_tensor = torch.argmax(one_hot, dim=-1)
                # encoding = self.vqvae._embedding(indices_tensor)
                # recon_loss = F.mse_loss(encoding, enc_embedding)
            else:
                # inference_com and AutoEncoding task.

                # id = 3
                # mask = srl_embed_ids == id
                # batch_size = encoding.shape[0]
                # mask = mask.unsqueeze(-1).expand((batch_size, self.set_seq_size, self.t5_model.config.d_model)).to(encoding.device)
                #
                # # feed the semantic role you want to disentangle into a linear layer (remove context information).
                # masked_input = torch.where(mask == True, encoding, torch.zeros_like(encoding))
                # remain_encoding = torch.where(mask == False, encoding, torch.zeros_like(encoding))
                # masked_output = self.enc_context_linear(masked_input)
                # encoding = remain_encoding + masked_output

                # ----------------------------------------------------------------------------------------------------
                # srl_encoding = self.t5_model.encoder.srl_embed(srl_embed_ids)
                # encoding += srl_encoding
                # ----------------------------------------------------------------------------------------------------
                recon_loss, encoding, perplexity, _ = self.vqvae(encoding, srl_ids)  # (batch_size, seq_len, x_dim)

                # ----------------------------------------------------------------------------------------------------
                # # Then, add context information back.
                # masked_input = torch.where(mask == True, encoding, torch.zeros_like(encoding))
                # remain_encoding = torch.where(mask == False, encoding, torch.zeros_like(encoding))
                # masked_output = self.dec_context_linear(masked_input)
                # encoding = remain_encoding + masked_output

        else:
            pass

        decoder_ce = self.decoder_loss(output_ids, label_ids, encoding, ignore_index=self.config.pad_token_id,
                                       encoder_attention_mask=attention_mask)

        return decoder_ce, recon_loss, reg_loss, loss_KL


if __name__ == '__main__':
    pass
