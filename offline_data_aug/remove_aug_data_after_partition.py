import json
from pathlib import Path

def streamline_partition(json_path):
    """
    读取 partition.json，保留 train 的全部内容，
    但删除 valid 和 test 中包含 _pitch_shift 或 _aug 的条目。
    """
    json_path = Path(json_path)
    
    if not json_path.exists():
        print(f"Error: 找不到文件 {json_path}")
        return

    # 1. 加载原始数据
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 记录原始数量以便对比
    stats = {k: len(v) for k, v in data.items()}
    print(f"处理前样本数量: {stats}")

    # 2. 核心逻辑：过滤 valid 和 test
    # 规则：如果名称中不含 '_pitch_shift' 且不含 '_aug'，则保留
    for split in ['valid', 'test']:
        if split in data:
            data[split] = [
                stem for stem in data[split] 
                if '_pitch_shift' not in stem and '_aug' not in stem
            ]

    # 3. 统计处理后数量
    new_stats = {k: len(v) for k, v in data.items()}
    print(f"处理后样本数量: {new_stats}")

    # 4. 写回文件 (覆盖原文件或另存为)
    # 建议先另存为一个新文件进行检查，确认没问题后再覆盖
    output_path = json_path.with_name(f"{json_path.stem}_streamlined.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"处理完成！新的划分文件已保存至: {output_path}")

if __name__ == "__main__":
    # 替换为你实际的 json 路径
    PARTITION_FILE = '/data/lym/F0_SOTA/penn/penn/assets/aug_partitions/ptdb.json' 
    streamline_partition(PARTITION_FILE)