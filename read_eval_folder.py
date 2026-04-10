import os
import json
import pandas as pd
import math

def safe_get_rpa(data, key):
    """安全读取 rpa，处理 NaN 和缺失情况"""
    try:
        value = data[key]["rpa"]
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return value
    except:
        return None

def collect_rpa(root_dir):
    results = []

    for root, dirs, files in os.walk(root_dir):
        if "overall.json" in files:
            json_path = os.path.join(root, "overall.json")

            # 子文件夹名称（你要的第一列）
            folder_name = os.path.basename(root)

            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"读取失败: {json_path}, error: {e}")
                continue

            mdb_rpa = safe_get_rpa(data, "mdb")
            ptdb_rpa = safe_get_rpa(data, "ptdb")
            agg_rpa = safe_get_rpa(data, "aggregate")

            results.append({
                "folder": folder_name,
                "mdb_rpa": mdb_rpa,
                "ptdb_rpa": ptdb_rpa,
                "agg_rpa": agg_rpa
            })

    return pd.DataFrame(results)


if __name__ == "__main__":
    root_dir = r"./eval"   # ← 修改这里

    df = collect_rpa(root_dir)

    # 保存为表格
    save_path = os.path.join(root_dir, "rpa_summary.csv")
    df.to_csv(save_path, index=False)

    print(df)
    print(f"\n已保存到: {save_path}")