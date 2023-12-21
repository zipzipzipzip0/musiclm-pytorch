### STEP 1: Train MuLaN ###

import torch
from musiclm_pytorch import MuLaN, AudioSpectrogramTransformer, TextTransformer

audio_transformer = AudioSpectrogramTransformer(
    dim = 512,
    depth = 6,
    heads = 8,
    dim_head = 64,
    spec_n_fft = 128,
    spec_win_length = 24,
    spec_aug_stretch_factor = 0.8
)

text_transformer = TextTransformer(
    dim = 512,
    depth = 6,
    heads = 8,
    dim_head = 64
)

mulan = MuLaN(
    audio_transformer = audio_transformer,
    text_transformer = text_transformer
)

# get a ton of <sound, text> pairs and train

wavs = torch.randn(2, 1024)
texts = torch.randint(0, 20000, (2, 256))

loss = mulan(wavs, texts)
loss.backward()

# after much training, you can embed sounds and text into a joint embedding space
# for conditioning the audio LM

embeds = mulan.get_audio_latents(wavs)  # during training

embeds = mulan.get_text_latents(texts)  # during inference


### STEP 2: Obtain conditioning embeddings ###

from musiclm_pytorch import MuLaNEmbedQuantizer

# setup the quantizer with the namespaced conditioning embeddings, unique per quantizer as well as namespace (per transformer)

quantizer = MuLaNEmbedQuantizer(
    mulan = mulan,                          # pass in trained mulan from above
    conditioning_dims = (1024, 1024, 1024), # say all three transformers have model dimensions of 1024
    namespaces = ('semantic', 'coarse', 'fine')
)

# now say you want the conditioning embeddings for semantic transformer

wavs = torch.randn(2, 1024)
conds = quantizer(wavs = wavs, namespace = 'semantic') # (2, 8, 1024) - 8 is number of quantizers


### STEP 3: Set up Soundstream and train the three transformers ###

from audiolm_pytorch import MusicLMSoundStream, SoundStreamTrainer

soundstream = MusicLMSoundStream()

trainer = SoundStreamTrainer(
    soundstream,
    folder = '/audio/soundstream',
    batch_size = 4,
    grad_accum_every = 8,
    data_max_length_seconds = 2,
    num_train_steps = 1_000_000
).cuda()

from audiolm_pytorch import HubertWithKmeans

wav2vec = HubertWithKmeans(
    checkpoint_path = './hubert/hubert_base_ls960.pt',
    kmeans_path = './hubert/hubert_base_ls960_L9_km500.bin'
)

from audiolm_pytorch import SemanticTransformer, SemanticTransformerTrainer

semantic_transformer = SemanticTransformer(
    num_semantic_tokens = wav2vec.codebook_size,
    dim = 1024,
    depth = 6,
    audio_text_condition = True      # this must be set to True (same for CoarseTransformer and FineTransformers)
).cuda()

trainer = SemanticTransformerTrainer(
    transformer = semantic_transformer,
    wav2vec = wav2vec,
    audio_conditioner = quantizer,   # pass in the MulanEmbedQuantizer instance above
    folder ='/audio/semantic',
    batch_size = 1,
    data_max_length = 320 * 32,
    num_train_steps = 1
)

trainer.train()

from audiolm_pytorch import CoarseTransformer, CoarseTransformerTrainer

coarse_transformer = CoarseTransformer(
    num_semantic_tokens = wav2vec.codebook_size,
    codebook_size = 1024,
    num_coarse_quantizers = 3,
    dim = 512,
    depth = 6,
    audio_text_condition = True
)

trainer = CoarseTransformerTrainer(
    transformer = coarse_transformer,
    codec = soundstream,
    wav2vec = wav2vec,
    folder = '/audio/coarse',
    batch_size = 1,
    data_max_length = 320 * 32,
    num_train_steps = 1_000_000
)

trainer.train()

from audiolm_pytorch import FineTransformer, FineTransformerTrainer

fine_transformer = FineTransformer(
    num_coarse_quantizers = 3,
    num_fine_quantizers = 5,
    codebook_size = 1024,
    dim = 512,
    depth = 6,
    audio_text_condition = True
)

trainer = FineTransformerTrainer(
    transformer = fine_transformer,
    codec = soundstream,
    folder = '/audio/fine',
    batch_size = 1,
    data_max_length = 320 * 32,
    num_train_steps = 1_000_000
)

trainer.train()


### STEP 4: Assemble AudioLM ###

from audiolm_pytorch import AudioLM

audio_lm = AudioLM(
    wav2vec = wav2vec,
    codec = soundstream,
    semantic_transformer = semantic_transformer,
    coarse_transformer = coarse_transformer,
    fine_transformer = fine_transformer
)


### STEP 5: Run model ###

from musiclm_pytorch import MusicLM

musiclm = MusicLM(
    audio_lm = audio_lm,                 # `AudioLM` from https://github.com/lucidrains/audiolm-pytorch
    mulan_embed_quantizer = quantizer    # the `MuLaNEmbedQuantizer` from above
)

music = musiclm('a fanfare style victory march', num_samples = 4) # sample 4 and pick the top match with mulan