import numpy as np
from audiomentations import Compose, AddGaussianNoise, SevenBandParametricEQ, HighPassFilter, Gain
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


import os
import json
import shutil
import soundfile as sf
import numpy as np
from pathlib import Path
from audiomentations import Compose, AddGaussianNoise, SevenBandParametricEQ, HighPassFilter, Gain
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 如果后续涉及 torch，也需在此添加 torch.manual_seed(seed)

set_seed(42)
class OfflineAugmenter:
    def __init__(self):
        # 定义概率触发管线：解决音色解耦与环境泛化
        self.pipeline = Compose([
            # 1. 频谱扭曲：破坏完美泛音分布 (p=0.8)
            SevenBandParametricEQ(min_gain_db=-6.0, max_gain_db=6.0, p=0.8),
            # 2. 随机滤波：模拟采集设备频响 (p=0.5)
            HighPassFilter(min_cutoff_freq=20, max_cutoff_freq=150, p=0.5),
            # 3. 环境加噪：模拟真实底噪 (p=0.4)
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.010, p=0.4),
            # 4. 幅度变换：模拟录音电平波动 (p=0.3)
            Gain(min_gain_db=-10.0, max_gain_db=0.0, p=0.3)
        ])

    def generate_augmented_variant(self, audio_path, label_path, output_audio_dir, output_f0_dir, output_json_dir):
        """
        核心函数：输入音频路径和标签路径，生成二次增强的三位一体文件包
        """
        # --- 1. 准备路径逻辑 ---
        audio_path = Path(audio_path)
        label_path = Path(label_path)
        

        base_name = audio_path.stem
        # 命名规则：原始名_aug.wav / .f0 / .json
        out_audio_p = output_audio_dir / f"{base_name}_aug.wav"
        out_label_p = output_f0_dir / f"{base_name}_aug.csv"
        out_json_p = output_json_dir / f"{base_name}_aug.json"

        # --- 2. 音频处理 ---
        with sf.SoundFile(audio_path) as f:
            sr = f.samplerate
            audio = f.read(dtype='float32')

        # 应用管线
        augmented_audio = self.pipeline(samples=audio, sample_rate=sr)

        # --- 3. 提取并提取参数记录 ---
        params_log = {}
        for transform in self.pipeline.transforms:
            t_name = transform.__class__.__name__
            params_log[t_name] = {
                "applied": transform.parameters.get("should_apply", False),
                "params": {k: v for k, v in transform.parameters.items() if k != "should_apply"}
            }

        # --- 4. 文件落地 ---
        # A. 保存音频 (强制 PCM_16)
        if np.max(np.abs(augmented_audio)) > 0.99:
            augmented_audio = augmented_audio / np.max(np.abs(augmented_audio))
        sf.write(out_audio_p, augmented_audio, sr, subtype='PCM_16')

        # B. 同步保存标签 (直接拷贝并重命名，因为二次增强不改音高)
        # 如果 label_path 存在则拷贝，否则打印警告
        if label_path.exists():
            shutil.copy(label_path, out_label_p)
        else:
            print(f"Warning: Label file {label_path} not found.")

        # C. 保存参数 JSON
        with open(out_json_p, 'w', encoding='utf-8') as j:
            json.dump(params_log, j, indent=4, ensure_ascii=False)

        return str(out_audio_p)

# --- 批量处理示例逻辑 ---
def run_batch_secondary_aug(src_audio_list, src_label_list, output_path):
    augmenter = OfflineAugmenter()
    for a_path, l_path in zip(src_audio_list, src_label_list):
        augmenter.generate_augmented_variant(a_path, l_path, output_path)

# --- 使用示例 ---
# augmenter = SecondaryAugmenter(sr=44100)
# audio_aug = augmenter.apply(audio_p) # audio_p 是你 Pitch Shift 后的音频

def ptdb():
    """Preprocessing ptdb dataset"""
    # Get audio files
    datadir = Path(r'./data/AUG_datasets')
    directory = datadir / 'ptdb' / 'SPEECH DATA'
    male = (directory / 'MALE' / 'MIC').rglob('*.wav')
    female = (directory / 'FEMALE' / 'MIC').rglob('*.wav')
    audio_files = sorted(itertools.chain(male, female))

    # Get pitch files
    pitch_files = [
        file.parent.parent.parent /
        'REF' /
        file.parent.name /
        file.with_suffix('.csv').name.replace('mic', 'ref')
        for file in audio_files]
    
    augmenter = OfflineAugmenter()
    for i, (audio_file, pitch_file) in enumerate(
        tqdm(zip(audio_files, pitch_files), total=len(audio_files), desc="Preprocessing ptdb")
    ):
              
        rel_audio = audio_file.relative_to(AUG_PTDB_DIR)
        rel_pitch = pitch_file.relative_to(AUG_PTDB_DIR)

        # === 2. 新路径（保持结构）===
        new_audio_dir = AUG_PTDB_DIR / rel_audio.parent
        new_f0_dir = AUG_PTDB_DIR / rel_pitch.parent
        output_json_dir = AUG_PTDB_DIR / 'AUG_PARAMS' / rel_audio.parent
        output_json_dir.mkdir(parents=True, exist_ok=True)
        # print({'audio': audio_file, 'pitch': pitch_file})
        # print({'new_audio_dir': new_audio_dir, 'new_f0_dir': new_f0_dir})
        augmenter.generate_augmented_variant(audio_file, pitch_file, new_audio_dir, new_f0_dir, output_json_dir)
        print(new_audio_dir, new_f0_dir, output_json_dir)
        # break
def mdb():
    """Preprocessing mdb dataset"""
    # Get audio files
    
    audio_files = AUG_MDB_AUDIO_DIR.glob('*.wav')
    audio_files = sorted([
        file for file in audio_files if not file.stem.startswith('._')])

    # Get pitch files
    pitch_files = [
        AUG_MDB_ANNOTATION_DIR /
        file.with_suffix('.csv').name
        for file in audio_files]
    # Get pitch files
    augmenter = OfflineAugmenter()
    for i, (audio_file, pitch_file) in enumerate(
        tqdm(zip(audio_files, pitch_files), total=len(audio_files), desc="Preprocessing ptdb")
    ):
              
        rel_audio = audio_file.relative_to(AUG_MDB_DIR)
        rel_pitch = pitch_file.relative_to(AUG_MDB_DIR)

        # === 2. 新路径（保持结构）===
        new_audio_dir = AUG_MDB_DIR / rel_audio.parent
        new_f0_dir = AUG_MDB_DIR / rel_pitch.parent
        output_json_dir = AUG_MDB_DIR / 'AUG_PARAMS' / rel_audio.parent
        output_json_dir.mkdir(parents=True, exist_ok=True)
        # print({'audio': audio_file, 'pitch': pitch_file})
        # print({'new_audio_dir': new_audio_dir, 'new_f0_dir': new_f0_dir})
        augmenter.generate_augmented_variant(audio_file, pitch_file, new_audio_dir, new_f0_dir, output_json_dir)
        print(new_audio_dir, new_f0_dir, output_json_dir)
        # break
if __name__ == "__main__":
    ptdb()
    # mdb()