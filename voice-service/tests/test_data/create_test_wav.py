"""Create test WAV file."""
import struct
from pathlib import Path

import numpy as np

sample_rate = 24000
duration = 0.5
samples = int(sample_rate * duration)

t = np.linspace(0, duration, samples, dtype=np.float32)
audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
pcm = (audio * 32767).astype(np.int16)

buffer = bytearray()
buffer.extend(b"RIFF")
buffer.extend(struct.pack("<I", 36 + len(pcm) * 2))
buffer.extend(b"WAVE")
buffer.extend(b"fmt ")
buffer.extend(struct.pack("<I", 16))
buffer.extend(struct.pack("<HHIIHH", 1, 1, sample_rate, sample_rate * 2, 2, 16))
buffer.extend(b"data")
buffer.extend(struct.pack("<I", len(pcm) * 2))
buffer.extend(pcm.tobytes())

out_dir = Path(__file__).parent
(out_dir / "test.wav").write_bytes(bytes(buffer))
(out_dir / "test.lab").write_text("Hello world")
print("Created test.wav and test.lab")



