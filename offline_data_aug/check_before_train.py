import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path

def check_alignment(audio_path, pitch_path, voiced_path, sr=16000, hop_size=128):
    """
    可视化音频时频谱与 F0 标签的对齐情况
    """
    # 1. 加载数据
    if str(audio_path).endswith('.npy'):
        audio = np.load(audio_path)
    else:
        audio, _ = librosa.load(audio_path, sr=sr)
        
    pitch = np.load(pitch_path)
    voiced = np.load(voiced_path)

    # 2. 计算时频谱 (STFT)
    # n_fft 建议取 hop_size 的 4 倍以上以获得较好的频率分辨率
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio, hop_length=hop_size)), ref=np.max)

    # 3. 准备绘制
    plt.figure(figsize=(12, 6))
    # plt.subplot(2,1,1)
    # 绘制背景时频谱
    librosa.display.specshow(D, sr=sr, hop_length=hop_size, x_axis='time', y_axis='log', cmap='magma')
    
    # 4. 准备 F0 曲线
    # 只绘制有声部分的音高，无声部分设为 NaN 绘图时会自动跳过
    plot_pitch = pitch.copy()
    plot_pitch[voiced == 0] = np.nan
    # plot_pitch = 440.0 * (2.0 ** ((plot_pitch - 69.0) / 12.0))
    
    # 构造时间轴
    times = librosa.times_like(plot_pitch, sr=sr, hop_length=hop_size)
    times = times/8
    # plt.subplot(2,1,2)
    # 5. 叠加 F0 曲线 (用亮绿色或青色以区分背景)
    plt.plot(times, plot_pitch, label='F0 Label', color='cyan', linewidth=2, alpha=0.8)
    
    plt.title(f"Alignment Check: {Path(audio_path).stem}")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.legend(loc='upper right')
    plt.colorbar(format='%+2.0f dB')
    
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{Path(audio_path).stem}_alignment_check.png")

if __name__ == "__main__":
    # 示例路径，替换为你 Cache 文件夹中的一个样本
    # CACHE_DIR = Path('./data/aug_cache/mdb')
    CACHE_DIR = Path('./data/aug_cache/ptdb')
    # STEM = "MusicDelta_Rock_STEM_02.RESYN"
    STEM = 'mic_F01_sa1_aug'
    # STEM = "000000"
    
    audio_file = CACHE_DIR / f"{STEM}.wav"
    pitch_file = CACHE_DIR / f"{STEM}-pitch.npy"
    voiced_file = CACHE_DIR / f"{STEM}-voiced.npy"
    print(audio_file, pitch_file, voiced_file)
    if audio_file.exists():
        # 注意：这里的 hop_size 必须与你预处理脚本中的一致 (MDB 128, PTDB 160)
        check_alignment(audio_file, pitch_file, voiced_file, sr=16000, hop_size=1024)
    else:
        print("文件不存在，请检查路径和 STEM 名称。")