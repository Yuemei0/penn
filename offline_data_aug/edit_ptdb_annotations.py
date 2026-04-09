from pathlib import Path
import itertools
import numpy as np
import pandas as pd


# PTDB analysis parameters
PTDB_HOPSIZE = 160  # samples
PTDB_SAMPLE_RATE = 16000  # samples per second
PTDB_WINDOW_SIZE = 512  # samples
PTDB_HOPSIZE_SECONDS = PTDB_HOPSIZE / PTDB_SAMPLE_RATE


datadir = Path(r'./data/datasets')
directory = datadir / 'ptdb' / 'SPEECH DATA'
male = (directory / 'MALE' / 'MIC').rglob('*.wav')
female = (directory / 'FEMALE' / 'MIC').rglob('*.wav')
audio_files = sorted(itertools.chain(male, female))

# Get pitch files
pitch_files = [
    file.parent.parent.parent /
    'REF' /
    file.parent.name /
    file.with_suffix('.f0').name.replace('mic', 'ref')
    for file in audio_files]

for pitch_file in pitch_files:
    pitch = np.loadtxt(open(pitch_file), delimiter=' ')[:, 0]
    times = PTDB_HOPSIZE_SECONDS * np.arange(0, len(pitch))
    times += PTDB_HOPSIZE_SECONDS / 2
    new_path = pitch_file.parent.parent.parent / \
                   'REF_NEW' / \
                   pitch_file.parent.name / \
                   pitch_file.with_suffix('.csv').name
    print(new_path)
    new_path.parent.mkdir(parents=True, exist_ok=True)
    
    assert len(times) == len(pitch)
    dataframe = pd.DataFrame({
        'time': times,
        'frequency': pitch
    })
    dataframe.to_csv(new_path, index=False, header=False)
    

