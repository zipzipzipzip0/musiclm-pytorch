from audiolm_pytorch import HubertWithKmeans
from audiolm_pytorch import CoarseTransformer, CoarseTransformerTrainer
import torch
from audiolm_pytorch import MusicLMSoundStream, SoundStreamTrainer
from musiclm_pytorch import MuLaNEmbedQuantizer
from musiclm_pytorch import MuLaN, AudioSpectrogramTransformer, TextTransformer


soundstream = MusicLMSoundStream(
    codebook_size = 4096
)

soundstream.load_state_dict(torch.load('saves/soundstream.pth'))

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

quantizer = MuLaNEmbedQuantizer(
    mulan = mulan,                          # pass in trained mulan from above
    conditioning_dims = (1024, 1024, 1024), # say all three transformers have model dimensions of 1024
    namespaces = ('semantic', 'coarse', 'fine')
)

dataset_path = './audio/fma_small'

wav2vec = HubertWithKmeans(
    checkpoint_path = './hubert/hubert_base_ls960.pt',
    kmeans_path = './hubert/hubert_base_ls960_L9_km500.bin'
)

coarse_transformer = CoarseTransformer(
    num_semantic_tokens = wav2vec.codebook_size,
    codebook_size = 1024,
    num_coarse_quantizers = 3,
    dim = 1024,
    depth = 6,
    audio_text_condition = True
)

def train_coarse():
    coarse_trainer = CoarseTransformerTrainer(
        transformer = coarse_transformer,
        codec = soundstream,
        wav2vec = wav2vec,
        audio_conditioner = quantizer,
        folder = dataset_path,
        batch_size = 1,
        data_max_length = 320 * 32,
        num_train_steps = 50_000
    ).cuda()

    coarse_trainer.train()
    del coarse_trainer

train_coarse()

torch.save(coarse_transformer.state_dict(), 'saves/coarse_transformer.pth')