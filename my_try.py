# import config.crepe++ as crepe_config
import penn
import torch
import torchaudio
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import sys 
import importlib
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 2. 动态导入 config/crepe++.py 模块
# 第一个参数是模块名（自定义，仅在当前脚本内使用），第二个是文件路径
crepe_config = importlib.machinery.SourceFileLoader(
    'crepe_config',  # 自定义模块别名
    '/data/lym/F0_SOTA/penn/runs/crepe++/crepe++.py'
    # '/data/lym/F0_SOTA/penn/runs/crepe/crepe.py'
    # '/data/lym/F0_SOTA/penn/runs/crepe++-mdb/crepe++-mdb.py'
).load_module()
penn.configure(crepe_config)
print(f"当前模型: {penn.MODEL}")
print(f"当前音高bins: {penn.PITCH_BINS}")

# checkpoint = '/data/lym/F0_SOTA/penn/runs/crepe++_distaware/00250000.pt'
# method_name = 'crepe++_distaware'
# checkpoint = r'/data/lym/F0_SOTA/penn-master/penn/fcnf0++.pt'
# checkpoint = r'/data/lym/F0_SOTA/penn/runs/fcnf0++-mdb/00250000.pt'
checkpoint = r'/data/lym/F0_SOTA/penn/runs/crepe++/00250000.pt'
# checkpoint = r'/data/lym/F0_SOTA/penn/runs/crepe/00118500.pt'
# checkpoint = r'/data/lym/F0_SOTA/penn/runs/crepe++-mdb/00250000.pt'
# checkpoint = r'/data/lym/F0_SOTA/penn/runs/crepe/00118500.pt'
# checkpoint = r'/data/lym/F0_SOTA/penn/runs/crepe++-ptdb/00050000.pt'

# method_name = 'fcnf0plusplus'
# audiopath = r'/data/lym/F0_DATASETS/Vocadito/Audio'
# audiopath = r'/data/lym/F0_DATASETS/MIR-1K/Wavfile'
# audiopath = r'/data/lym/F0_DATASETS/test_audio_and_res/data/output_resample_mir1k'
# respath = r'/data/lym/F0_DATASETS/test_audio_and_res/result/output_resample_mir1k/fcnf0plusplus/'

# # method_name = 'crepeplusplus'
# audiopath=r'/data/lym/F0_DATASETS/test_audio_and_res/data/sweep'
# respath = r'/data/lym/F0_DATASETS/test_audio_and_res/result/sweep/crepepp-ptdb50k/'

# audiopath = r'/data/lym/F0_DATASETS/test_audio_and_res/data/output_resample_ptdb'
# respath = r'/data/lym/F0_DATASETS/test_audio_and_res/result/output_resample_ptdb/crepepp/'

audiopath = r'/data/lym/F0_DATASETS/test_audio_and_res/data/output_resample_gamelan'
respath = r'/data/lym/F0_DATASETS/test_audio_and_res/result/output_resample_gamelan/crepepp/'

os.makedirs(respath, exist_ok=True)
files = os.listdir(audiopath)

hopsize = .01
fmin = 30.
fmax = 1200.
gpu = 1
batch_size = 2048
center = 'half-hop'
interp_unvoiced_at = .065

# penn.DECODER = 'local_expected_value'
# penn.DECODER = 'viterbi'
# penn.DECODER = 'argmax'

for file in tqdm(files, desc="Processing audio files"):
    filepath = os.path.join(audiopath, file)
    audio, sample_rate = torchaudio.load(filepath)
    if audio.shape[0] == 2:
        audio = audio[1:2, :] 

    pitch, periodicity = penn.from_audio(
        audio,
        sample_rate=sample_rate,
        hopsize=hopsize,
        fmin=fmin,
        fmax=fmax,
        checkpoint=checkpoint,
        batch_size=batch_size,
        center=center,
        interp_unvoiced_at=interp_unvoiced_at,
        gpu=gpu)

    pitch = pitch.detach().cpu().numpy().squeeze()
    periodicity = periodicity.detach().cpu().numpy().squeeze()

    time = hopsize / 2.0 + hopsize * np.arange(len(pitch))

    dataframe = pd.DataFrame({
        'time': time,
        'frequency': pitch,
        'confidence': periodicity
    })

    resname = respath + file[:-4] + '.csv'
    print(resname)
    dataframe.to_csv(resname, index=False)


# inference on multiple folders
# audio_root = r"/data/lym/F0_DATASETS/vocadito_shift"
# output_root = r"/data/lym/F0_RESULTS/vocadito_shift/fcnf0plusplus"
# semitone_range = np.arange(-0.5, 0.5, 0.1)  # 和你之前完全一致
# # ==========================================================
# hopsize = .01
# fmin = 30.
# fmax = 1200.
# gpu = 1
# batch_size = 2048
# center = 'half-hop'
# interp_unvoiced_at = .065
# # 遍历每个偏移量
# for shift in semitone_range:
#     shift_str = f"{shift:+.1f}st"
#     print(f"\n====== CREPE 预测：{shift_str} ======")

#     # 音频文件夹
#     audio_dir = os.path.join(audio_root, f"Audio_shifted_{shift_str}")
#     # 输出预测文件夹
#     out_dir = os.path.join(output_root, f"fcnf0plusplus_predictions_{shift_str}/")
#     os.makedirs(out_dir, exist_ok=True)

#     if not os.path.exists(audio_dir):
#         print(f"跳过：{audio_dir} 不存在")
#         continue

#     # 遍历该文件夹下所有音频
#     for filename in os.listdir(audio_dir):
#         if not filename.lower().endswith(('.wav', '.mp3', '.flac')):
#             continue

#         audio_path = os.path.join(audio_dir, filename)
#         name = os.path.splitext(filename)[0]
#         audio, sample_rate = torchaudio.load(audio_path)
#         pitch, periodicity = penn.from_audio(
#             audio,
#             sample_rate=sample_rate,
#             hopsize=hopsize,
#             fmin=fmin,
#             fmax=fmax,
#             checkpoint=checkpoint,
#             batch_size=batch_size,
#             center=center,
#             interp_unvoiced_at=interp_unvoiced_at,
#             gpu=gpu)

#         pitch = pitch.detach().cpu().numpy().squeeze()
#         periodicity = periodicity.detach().cpu().numpy().squeeze()

#         time = hopsize / 2.0 + hopsize * np.arange(len(pitch))

#         dataframe = pd.DataFrame({
#             'time': time,
#             'frequency': pitch,
#             'confidence': periodicity
#         })

#         resname = out_dir + filename[:-4] + '.csv'
#         print(resname)
#         dataframe.to_csv(resname, index=False)