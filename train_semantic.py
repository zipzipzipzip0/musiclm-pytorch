from audiolm_pytorch import HubertWithKmeans
from audiolm_pytorch import SemanticTransformer, SemanticTransformerTrainer
import torch
from musiclm_pytorch import MuLaNEmbedQuantizer
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

semantic_transformer = SemanticTransformer(
    num_semantic_tokens = wav2vec.codebook_size,
    dim = 1024,
    depth = 6,
    audio_text_condition = True      # this must be set to True (same for CoarseTransformer and FineTransformers)
)

def train_semantic():
    semantic_trainer = SemanticTransformerTrainer(
        transformer = semantic_transformer,
        wav2vec = wav2vec,
        audio_conditioner = quantizer,   # pass in the MulanEmbedQuantizer instance above
        folder = dataset_path,
        batch_size = 1,
        data_max_length = 320 * 32,
        num_train_steps = 50_000
    ).cuda()

    semantic_trainer.train()
    del semantic_trainer

train_semantic()

torch.save(semantic_transformer.state_dict(), 'saves/semantic_transformer.pth')