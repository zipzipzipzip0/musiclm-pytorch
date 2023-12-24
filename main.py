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

mulan.load_state_dict(torch.load('saves/mulan.pth'))

### STEP 2: Obtain conditioning embeddings ###

from musiclm_pytorch import MuLaNEmbedQuantizer

# setup the quantizer with the namespaced conditioning embeddings, unique per quantizer as well as namespace (per transformer)

quantizer = MuLaNEmbedQuantizer(
    mulan = mulan,                          # pass in trained mulan from above
    conditioning_dims = (1024, 1024, 1024), # say all three transformers have model dimensions of 1024
    namespaces = ('semantic', 'coarse', 'fine')
)

### STEP 3: Set up Soundstream and train the three transformers ###

from audiolm_pytorch import MusicLMSoundStream

dataset_path = './audio/fma_small'

soundstream = MusicLMSoundStream(
    codebook_size = 4096
)

soundstream.load_state_dict(torch.load('saves/soundstream.pth'))

from audiolm_pytorch import HubertWithKmeans

wav2vec = HubertWithKmeans(
    checkpoint_path = './hubert/hubert_base_ls960.pt',
    kmeans_path = './hubert/hubert_base_ls960_L9_km500.bin'
)

from audiolm_pytorch import SemanticTransformer

semantic_transformer = SemanticTransformer(
    num_semantic_tokens = wav2vec.codebook_size,
    dim = 1024,
    depth = 6,
    audio_text_condition = True      # this must be set to True (same for CoarseTransformer and FineTransformers)
)

semantic_transformer.load_state_dict(torch.load('saves/semantic_transformer.pth'))

from audiolm_pytorch import CoarseTransformer

coarse_transformer = CoarseTransformer(
    num_semantic_tokens = wav2vec.codebook_size,
    codebook_size = 1024,
    num_coarse_quantizers = 3,
    dim = 1024,
    depth = 6,
    audio_text_condition = True
)

coarse_transformer.load_state_dict(torch.load('saves/coarse_transformer.pth'))

from audiolm_pytorch import FineTransformer

fine_transformer = FineTransformer(
    num_coarse_quantizers = 3,
    num_fine_quantizers = 5,
    codebook_size = 1024,
    dim = 1024,
    depth = 6,
    audio_text_condition = True
)

fine_transformer.load_state_dict(torch.load('saves/fine_transformer.pth'))

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

prompt = 'a fanfare style victory march'

music = musiclm(prompt, num_samples = 4) # sample 4 and pick the top match with mulan

import torchaudio

output_path = f'./output/{prompt}.wav'
sample_rate = 44100
torchaudio.save(output_path, music.cpu(), sample_rate)