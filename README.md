### Improving Semantic Control in Discrete Latent Spaces with Transformer Quantized Variational Autoencoders

#### Training model:
```python
train_t5.py

# model_dict (line 225)
t5_model_name = 't5-small', 't5-base', 'google/flan-t5-base'

ae_latent_size =
  256, # t5-small
  768 # t5-base

latent_type =
  'T5_original', # vanilla T5
  'T5_vqvae' # our architecture

disentangled_vqvae = False # if true, different codebooks for different semantic roles.
ema = True

# data_dict (line 238)
task =
  'recon', # explanation recon
  'math_recon', # math expression recon
  'math_inf' # math derivation,
  'inference_com' # syllogistic deductive NLI (currently, the annotation is unavailable online)
```
