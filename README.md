EACL2024 Finding: Improving Semantic Control in Discrete Latent Spaces with Transformer Quantized Variational Autoencoders [link](https://arxiv.org/abs/2402.00723#:~:text=Experimental%20results%20indicate%20that%20T5VQVAE,%2C%20text%20transfer%2C%20and%20inference.)
```latex
@inproceedings{zhang-etal-2024-improving,
    title = "Improving Semantic Control in Discrete Latent Spaces with Transformer Quantized Variational Autoencoders",
    author = "Zhang, Yingji  and
      Carvalho, Danilo  and
      Valentino, Marco  and
      Pratt-Hartmann, Ian  and
      Freitas, Andre",
    editor = "Graham, Yvette  and
      Purver, Matthew",
    booktitle = "Findings of the Association for Computational Linguistics: EACL 2024",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-eacl.97/",
    pages = "1434--1450",
    abstract = "Achieving precise semantic control over the latent spaces of Variational AutoEncoders (VAEs) holds significant value for downstream tasks in NLP as the underlying generative mechanisms could be better localised, explained and improved upon. Recent research, however, has struggled to achieve consistent results, primarily due to the inevitable loss of semantic information in the variational bottleneck and limited control over the decoding mechanism. To overcome these challenges, we investigate discrete latent spaces in Vector Quantized Variational AutoEncoder (VQVAE) to improve semantic control and generation in Transformer-based VAEs. In particular, We propose T5VQVAE, a novel model that leverages the controllability of VQVAE to guide the self-attention mechanism in T5, exploiting its full generalization capabilities. Experimental results indicate that T5VQVAE outperforms existing state-of-the-art VAE models, including Optimus, in terms of control and preservation of semantic information across different tasks such as auto-encoding of sentences and mathematical expressions, text transfer, and inference. Moreover, T5VQVAE exhibits improved reasoning capabilities, suggesting potential applications for downstream natural language and symbolic inference tasks."
}
```
#### Training/evaluating model:
```python
train_t5.py

# model_dict (line 225)
t5_model_name = 't5-base'

ae_latent_size = 768

latent_type = 'T5_vqvae'

disentangled_vqvae = False # if true, different codebooks for different semantic roles.
ema = True

# data_dict (line 238)
train_data_file = 'datasets/full/WorldTree/explanations_tr.txt'
test_data_file = 'datasets/full/WorldTree/explanations_te.txt'
task = 'recon'
```
#### Checkpoint:

[checkpoints for T5VQVAE(base)](https://drive.google.com/drive/folders/1RfKr1RvMXuaJICzjYKqqpnaeAU_rCH3W?usp=sharing)
