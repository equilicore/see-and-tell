import pytest
import numpy as np
import torch
import torchaudio as ta
import os

from cntell.listen.speech import Listen
from cntell.say.caption import Say


@pytest.fixture()
def random_audio():
    say = Say()
    say.prepare()
    out = say.run(Say.Caption(text="Hello world! This is a test."))
    
    # Transform mono to stereo
    out.audio = np.repeat(out.audio, 2).reshape(-1, 2).T
    ta.save('test.wav', torch.tensor(out.audio), 16000)
    
    yield 'test.wav'
    
    os.remove('test.wav')z


def test_listen_prepare():
    listen = Listen()
    listen.prepare()
    
    
def test_listen_run(random_audio):
    listen = Listen()
    listen.prepare()
    
    # 'Assemble-on-fly' style
    listen.run(path=random_audio)
    
    # Explicit style
    out = listen.run(Listen.PathToAudio(path=random_audio))
    assert out.segments is not None