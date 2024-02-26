EACL2024 Finding: Improving Semantic Control in Discrete Latent Spaces with Transformer Quantized Variational Autoencoders [link](https://arxiv.org/abs/2402.00723#:~:text=Experimental%20results%20indicate%20that%20T5VQVAE,%2C%20text%20transfer%2C%20and%20inference.)
```latex
@inproceedings{Zhang2024ImprovingSC,
  title={Improving Semantic Control in Discrete Latent Spaces with Transformer Quantized Variational Autoencoders},
  author={Yingji Zhang and Danilo S. Carvalho and Marco Valentino and Ian Pratt-Hartmann and Andr'e Freitas},
  year={2024},
  url={https://api.semanticscholar.org/CorpusID:267365107}
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
