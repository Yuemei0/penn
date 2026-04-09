# 将mdb,ptdb数据做增强，并按照原始结构保存，以便适配penn框架

import numpy as np
import random
import soxr
import librosa
import soundfile as sf
from pathlib import Path
import itertools
import pandas as pd
from tqdm import tqdm

SEED = 42

DATA_DIR = Path(r'./data/datasets')

PTDB_DIR = DATA_DIR / 'ptdb'
PTDB_AUDIO_DIR = PTDB_DIR / 'SPEECH DATA'
PTDB_REF_DIR = PTDB_DIR / 'REF_NEW'

MDB_DIR = DATA_DIR / 'mdb'
MDB_AUDIO_DIR = MDB_DIR / 'audio_stems'
MDB_ANNOTATION_DIR = MDB_DIR / 'annotation_stems'

AUG_DATA_DIR = Path(r'./data/AUG_datasets')

AUG_PTDB_DIR = AUG_DATA_DIR / 'ptdb'
AUG_PTDB_AUDIO_DIR = AUG_PTDB_DIR / 'SPEECH DATA'
AUG_PTDB_REF_DIR = AUG_PTDB_DIR / 'REF'

AUG_MDB_DIR = AUG_DATA_DIR / 'mdb'
AUG_MDB_AUDIO_DIR = AUG_MDB_DIR / 'audio_stems'
AUG_MDB_ANNOTATION_DIR = AUG_MDB_DIR / 'annotation_stems'


# def micro_pitch_shift_resample(audio_path, f0_path, output_audio_dir,output_f0_dir, max_cents=50, seed=None):
#     """
#     高精度微音分搬移：基于时域重采样实现。

#     Args:
#         audio_path (str): 原始音频文件路径
#         f0_path (str): 原始 f0 标签文件路径（假设每行一个音高数值）
#         output_dir (str or Path): 保存增强后音频和标注的目录
#         max_cents (int): 最大偏移分位数

#     Returns:
#         output_audio_path (Path): 输出音频文件路径
#         output_f0_path (Path): 输出 F0 标签文件路径
#         cents_offset (float): 实际偏移的随机分位数
#     """
#     if seed is not None:
#         random.seed(seed)       # 固定 Python random
#         np.random.seed(seed) 
#     output_audio_dir.mkdir(parents=True, exist_ok=True)
#     output_f0_dir.mkdir(parents=True, exist_ok=True)
#     audio, sr = librosa.load(audio_path, sr=None)
#     # 1. 生成高精度随机偏移量 (不进行四舍五入)
#     cents_offset = random.uniform(-max_cents, max_cents)

#     # 2. 计算物理缩放比例 r
#     # r > 1 表示音高升高，时长缩短；r < 1 表示音高降低，时长增加
#     ratio = 2.0 ** (cents_offset / 1200.0)

#     # 3. 高精度重采样音频
#     # 原理：以 sr / ratio 的采样率读取音频，再重采样回 sr，实现频率搬移
#     # 使用 soxr 的 VHQ (Very High Quality) 模式
#     shifted_audio = soxr.resample(audio, sr, sr / ratio, quality='vhq')

#     # 4. 同步更新 F0 标签
#     annotations = np.loadtxt(open(f0_path), delimiter=',')
#     times, f0_orig = annotations[:, 0], annotations[:, 1]

#     # F0 的数值直接线性缩放
#     shifted_f0_values = f0_orig * ratio
#     shifted_times = times / ratio  
    

    
#     # 6. 写文件输出
#     audio_stem = Path(audio_path).stem
#     f0_suffix = '.csv'
    
#     # 输入 cents_offset = -12.345 → 输出 'm12d345';输入 cents_offset = +25.678 → 输出 'p25d678'
#     cents_tag = f"{cents_offset:+.3f}".replace('+', 'p').replace('-', 'm').replace('.', 'd')

#     output_audio_path = output_audio_dir / f"{audio_stem}_pitch_shift_{cents_tag}.wav"
#     output_f0_path = output_f0_dir / f"{audio_stem}_pitch_shift_{cents_tag}{f0_suffix}"
    
#     # print(output_audio_path, output_f0_path)
#     sf.write(output_audio_path, shifted_audio, sr)
#     assert len(shifted_times) == len(shifted_f0_values)
#     dataframe = pd.DataFrame({
#         'time': shifted_times,
#         'frequency': shifted_f0_values
#     })
#     dataframe.to_csv(output_f0_path, index=False, header=False)

    # return output_audio_path, output_f0_path, cents_offset
# 在 micro_pitch_shift_resample 中修改 seed 逻辑


def micro_pitch_shift_resample(audio_path, f0_path, output_audio_dir, output_f0_dir, max_cents=50, seed=None, local_seed_offset=0):
    """
    高精度微音分搬移：基于时域重采样实现，每个文件多次生成不同版本。
    
    seed: 全局随机种子
    local_seed_offset: 用于生成同一文件的多个版本
    """
    # 使用全局种子 + 文件名 hash + offset 生成局部随机器
    if seed is not None:
        file_stem = Path(audio_path).stem
        local_seed = hash((seed, file_stem, local_seed_offset)) % (2**32)
        rnd = random.Random(local_seed)    # 独立随机器
        np_random_state = np.random.RandomState(local_seed)
    else:
        rnd = random
        np_random_state = np.random

    output_audio_dir.mkdir(parents=True, exist_ok=True)
    output_f0_dir.mkdir(parents=True, exist_ok=True)
    audio, sr = librosa.load(audio_path, sr=None)

    # 生成随机偏移
    cents_offset = rnd.uniform(-max_cents, max_cents)

    # 计算音高搬移比例
    ratio = 2.0 ** (cents_offset / 1200.0)
    shifted_audio = soxr.resample(audio, sr, sr / ratio, quality='vhq')

    annotations = np.loadtxt(open(f0_path), delimiter=',')
    times, f0_orig = annotations[:, 0], annotations[:, 1]
    shifted_f0_values = f0_orig * ratio
    shifted_times = times / ratio  

    f0_suffix = '.csv'
    cents_tag = f"{cents_offset:+.3f}".replace('+', 'p').replace('-', 'm').replace('.', 'd')
    audio_stem = Path(audio_path).stem
    output_audio_path = output_audio_dir / f"{audio_stem}_pitch_shift_{cents_tag}.wav"
    output_f0_path = output_f0_dir / f"{audio_stem}_pitch_shift_{cents_tag}{f0_suffix}"

    sf.write(output_audio_path, shifted_audio, sr)
    dataframe = pd.DataFrame({'time': shifted_times, 'frequency': shifted_f0_values})
    dataframe.to_csv(output_f0_path, index=False, header=False)

    return output_audio_path, output_f0_path, cents_offset

def ptdb():
    """Preprocessing ptdb dataset"""
    # Get audio files
    datadir = Path(r'./data/datasets')
    directory = datadir / 'ptdb' / 'SPEECH DATA'
    male = (directory / 'MALE' / 'MIC').rglob('*.wav')
    female = (directory / 'FEMALE' / 'MIC').rglob('*.wav')
    audio_files = sorted(itertools.chain(male, female))

    # Get pitch files
    pitch_files = [
        file.parent.parent.parent /
        'REF_NEW' /
        file.parent.name /
        file.with_suffix('.csv').name.replace('mic', 'ref')
        for file in audio_files]
    
    for i, (audio_file, pitch_file) in enumerate(
        tqdm(zip(audio_files, pitch_files), total=len(audio_files), desc="Preprocessing ptdb")
    ):
              
        rel_audio = audio_file.relative_to(PTDB_DIR)
        rel_pitch = pitch_file.relative_to(PTDB_DIR)

        # === 2. 新路径（保持结构）===
        new_audio_dir = AUG_PTDB_DIR / rel_audio.parent
        new_f0_dir = AUG_PTDB_DIR / rel_pitch.parent
        # print({'audio': audio_file, 'pitch': pitch_file})
        # print({'new_audio_dir': new_audio_dir, 'new_f0_dir': new_f0_dir})
        for version in range(4):
            micro_pitch_shift_resample(audio_file, pitch_file, new_audio_dir, new_f0_dir, seed=SEED, local_seed_offset=version)
        # break
        
def mdb():
    """Preprocessing mdb dataset"""
    # Get audio files
    
    audio_files = MDB_AUDIO_DIR.glob('*.wav')
    audio_files = sorted([
        file for file in audio_files if not file.stem.startswith('._')])

    # Get pitch files
    pitch_files = [
        MDB_ANNOTATION_DIR /
        file.with_suffix('.csv').name
        for file in audio_files]
    # Get pitch files
    
    for i, (audio_file, pitch_file) in enumerate(
        tqdm(zip(audio_files, pitch_files), total=len(audio_files), desc="Preprocessing ptdb")
    ):
              
        rel_audio = audio_file.relative_to(MDB_DIR)
        rel_pitch = pitch_file.relative_to(MDB_DIR)

        # === 2. 新路径（保持结构）===
        new_audio_dir = AUG_MDB_DIR / rel_audio.parent
        new_f0_dir = AUG_MDB_DIR / rel_pitch.parent
        # print({'audio': audio_file, 'pitch': pitch_file})
        # print({'new_audio_dir': new_audio_dir, 'new_f0_dir': new_f0_dir})
        for version in range(4):
            micro_pitch_shift_resample(audio_file, pitch_file, new_audio_dir, new_f0_dir, seed=SEED, local_seed_offset=version)

if __name__ == "__main__":
    mdb()
    ptdb()