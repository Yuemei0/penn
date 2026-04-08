import json
import random

import penn


###############################################################################
# Dataset-specific
###############################################################################


def datasets(datasets):
    """Partition datasets"""
    for name in datasets:
        dataset(name)


def dataset(name):
    """Partition dataset"""
    # Get dataset stems
    stems = sorted([
        file.stem[:-6] for file in
        (penn.CACHE_DIR / name).glob('*-audio.npy')])
    
    # Separate original and augmented stems
    # Assuming augmented stems contain '_aug' in their name
    original_stems = [s for s in stems if '_aug' not in s]
    aug_stems = [s for s in stems if '_aug' in s]
    
    # Shuffle only original stems to determine splits
    random.seed(penn.RANDOM_SEED)
    random.shuffle(original_stems)

    # Get split points based on original stems
    left, right = int(.70 * len(original_stems)), int(.85 * len(original_stems))

    # Perform partition on original stems
    partition = {
        'train': sorted(original_stems[:left]),
        'valid': sorted(original_stems[left:right]),
        'test': sorted(original_stems[right:])}

    # Create mapping from original stem to its split
    source_to_split = {}
    for split, stems_list in partition.items():
        for stem in stems_list:
            source_to_split[stem] = split

    # Assign augmented stems to the same split as their source
    for aug_stem in aug_stems:
        # Extract source stem (assuming format: source_augX)
        source = aug_stem.split('_aug')[0]
        if source in source_to_split:
            split = source_to_split[source]
            partition[split].append(aug_stem)
        else:
            # If source not found, you might want to handle this case
            # For now, add to train or raise an error
            print(f"Warning: Source '{source}' for augmented stem '{aug_stem}' not found in original stems. Adding to train.")
            partition['train'].append(aug_stem)

    # Sort each split again
    for split in partition:
        partition[split].sort()

    # Write partition file
    with open(penn.PARTITION_DIR / f'{name}.json', 'w') as file:
        json.dump(partition, file, indent=4)
