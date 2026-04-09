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
    """Partition dataset by identifying root stems first"""
    # 1. 获取所有特征文件的 stem
    all_stems = sorted([
        file.stem[:-6] for file in
        (penn.CACHE_DIR / name).glob('*-audio.npy')])
    
    # 2. 寻找核心母本数据：不含 _pitch_shift 也不含 _aug
    # 逻辑：只有最原始的样本会被选入这个列表进行划分
    root_stems = [
        s for s in all_stems 
        if '_pitch_shift' not in s and '_aug' not in s
    ]
    
    # 3. 随机划分核心母本
    random.seed(penn.RANDOM_SEED)
    random.shuffle(root_stems)

    left, right = int(.70 * len(root_stems)), int(.85 * len(root_stems))
    
    # 建立母本到类别的映射
    root_to_split = {}
    for i, root in enumerate(root_stems):
        if i < left:
            root_to_split[root] = 'train'
        elif i < right:
            root_to_split[root] = 'valid'
        else:
            root_to_split[root] = 'test'

    # 4. 遍历所有数据，按前缀匹配母本
    partition = {'train': [], 'valid': [], 'test': []}
    
    # 将 root_stems 按长度降序排序，防止短前缀错误匹配长前缀（如 stem_1 匹配到 stem_10）
    # 虽然在你的命名规则下不一定发生，但这是严谨的做法
    sorted_roots = sorted(root_stems, key=len, reverse=True)

    for s in all_stems:
        # 寻找该 stem 属于哪个母本
        found_match = False
        for root in sorted_roots:
            if s.startswith(root):
                target_split = root_to_split[root]
                partition[target_split].append(s)
                found_match = True
                break
        
        if not found_match:
            # 兜底逻辑：处理没有找到母本的数据（例如母本被意外删除但衍生版还在）
            print(f"Warning: No root found for {s}. Adding to train.")
            partition['train'].append(s)

    # 5. 排序并写入 JSON
    for split in partition:
        partition[split].sort()

    with open(penn.PARTITION_DIR / f'{name}.json', 'w') as file:
        json.dump(partition, file, indent=4)
