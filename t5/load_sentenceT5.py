from sentence_transformers import SentenceTransformer
import copy


def load_sentenceT5_weight(t5_ae, sent_t5):
    new_state_dict = dict()
    for k, v in sent_t5.named_parameters():
        if k == '0.auto_model.shared.weight':
            new_state_dict[k] = copy.deepcopy(v)
        else:
            k = k.replace("0.auto_model.", "t5_model.")
            new_state_dict[k] = copy.deepcopy(v)

    new_t5_ae_stat_dict = dict()
    for k, v in t5_ae.named_parameters():
        if k in new_state_dict:
            new_t5_ae_stat_dict[k] = copy.deepcopy(new_state_dict[k])
        else:
            new_t5_ae_stat_dict[k] = copy.deepcopy(v)

    new_t5_ae_stat_dict['ae.encoder.dense.linear.weight'] = copy.deepcopy(new_state_dict['2.linear.weight'])
    # this is encoder embedding weights.
    new_t5_ae_stat_dict['enc_embed_weight.weight'] = copy.deepcopy(new_state_dict['0.auto_model.shared.weight'])

    # encoder, decoder, lm_head share the same weights.
    new_t5_ae_stat_dict['t5_model.lm_head.weight'] = new_t5_ae_stat_dict['t5_model.shared.weight']
    new_t5_ae_stat_dict['t5_model.encoder.embed_tokens.weight'] = new_t5_ae_stat_dict['t5_model.shared.weight']
    new_t5_ae_stat_dict['t5_model.decoder.embed_tokens.weight'] = new_t5_ae_stat_dict['t5_model.shared.weight']

    return new_t5_ae_stat_dict


if __name__ == '__main__':
    sentences = ["This is an example sentence", "Each sentence is converted"]
    model = SentenceTransformer('sentence-transformers/sentence-t5-base')
    embeddings = model.encode(sentences)
    print(embeddings)