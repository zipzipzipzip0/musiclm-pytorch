from audiolm_pytorch import FineTransformer, FineTransformerTrainer
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


fine_transformer = FineTransformer(
    num_coarse_quantizers = 3,
    num_fine_quantizers = 5,
    codebook_size = 1024,
    dim = 1024,
    depth = 6,
    audio_text_condition = True
)

def train_fine():
    fine_trainer = FineTransformerTrainer(
        transformer = fine_transformer,
        audio_conditioner = quantizer,
        codec = soundstream,
        folder = dataset_path,
        batch_size = 1,
        data_max_length = 320 * 32,
        num_train_steps = 50_000
    ).cuda()

    fine_trainer.train()
    del fine_trainer

train_fine()

torch.save(fine_transformer.state_dict(), 'saves/fine_transformer.pth')