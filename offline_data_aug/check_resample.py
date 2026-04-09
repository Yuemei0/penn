import numpy as np
import pandas as pd
import random
from pathlib import Path
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soxr

from step1_pitch_shift import micro_pitch_shift_resample

# -----------------------------
# F0 提取函数
# -----------------------------
def extract_f0(audio_path, sr=None, fmin=50.0, fmax=1000.0, hop_length=160):
    y, sr = librosa.load(audio_path, sr=sr)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length
    )
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
    return times, f0

# -----------------------------
# 测试脚本
# -----------------------------
def test_micro_pitch_shift():
    sr = 16000
    duration = 1.0  # 秒
    freq = 220.0  # Hz, A3
    t = np.linspace(0, duration, int(sr*duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * freq * t)

    # 保存原始音频和F0
    orig_audio_dir = Path("test_audio")
    orig_f0_dir = Path("test_f0")
    orig_audio_dir.mkdir(exist_ok=True)
    orig_f0_dir.mkdir(exist_ok=True)

    audio_path = orig_audio_dir / "sine.wav"
    f0_path = orig_f0_dir / "sine.csv"
    sf.write(audio_path, audio, sr)

    times = t
    f0_values = np.full_like(times, freq)
    pd.DataFrame({'time': times, 'frequency': f0_values}).to_csv(f0_path, index=False, header=False)

    # 输出目录
    output_audio_dir = Path("./offline_data_aug/aug_audio")
    output_f0_dir = Path("./offline_data_aug/aug_f0")
    output_audio_dir.mkdir(exist_ok=True)
    output_f0_dir.mkdir(exist_ok=True)
    
    # 微音分搬移
    shifted_audio_path, shifted_f0_path, cents_offset = micro_pitch_shift_resample(
        audio_path, f0_path, output_audio_dir, output_f0_dir, max_cents=50
    )
    print(f"Cents offset applied: {cents_offset:.3f}")

    # 提取搬移后的F0
    times_shifted, f0_shifted = extract_f0(shifted_audio_path)

    # 可视化对比
    plt.figure(figsize=(10,4))
    plt.plot(times, f0_values, label='Original F0')
    plt.plot(times_shifted, f0_shifted, label='Shifted F0', alpha=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Micro-Pitch Shift Test")
    plt.legend()
    # plt.show()
    plt.savefig("./offline_data_aug/micro_pitch_shift_test.png")

# 运行测试
test_micro_pitch_shift()