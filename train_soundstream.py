from audiolm_pytorch import MusicLMSoundStream, SoundStreamTrainer
import torch

dataset_path = './audio/fma_small'

soundstream = MusicLMSoundStream(
    codebook_size = 4096
)

def train_soundstream():
    soundstream_trainer = SoundStreamTrainer(
        soundstream,
        folder = dataset_path,
        batch_size = 4,
        grad_accum_every = 8,
        data_max_length_seconds = 2,
        num_train_steps = 1_000
    ).cuda()

    soundstream_trainer.train()
    del soundstream_trainer

train_soundstream()

torch.save(soundstream.state_dict(), 'saves/soundstream.pth')