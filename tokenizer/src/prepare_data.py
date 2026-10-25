from pathlib import Path
import random
import csv
from config import *

# 路径配置
BASE_DIR = Path(__file__).resolve().parents[2]
ZH_PATH = BASE_DIR / "data" / "raw" / "MultiUN.en-zh.zh"
EN_PATH = BASE_DIR / "data" / "raw" / "MultiUN.en-zh.en"
OUTPUT_DIR = BASE_DIR / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 参数配置
RANDOM_SEED = config['random_seed']
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.98, 0.01, 0.01
MAX_ZH_CHARS, MAX_EN_CHARS = 200, 300


def read_parallel_corpus(zh_path, en_path):
    """读取中英平行语料并校验行数一致。"""
    zh_lines = Path(zh_path).read_text(encoding="utf-8").splitlines()
    en_lines = Path(en_path).read_text(encoding="utf-8").splitlines()
    if len(zh_lines) != len(en_lines):
        raise ValueError(f"中英文行数不一致：{len(zh_lines)} vs {len(en_lines)}")
    print(f"原始句对数：{len(zh_lines):,}")
    return list(zip(zh_lines, en_lines))


def clean_pairs(pairs):
    """删除空句对、过长句对和重复句对。"""
    cleaned, seen = [], set()
    empty = long_ = dup = 0
    for zh, en in pairs:
        zh, en = zh.strip(), en.strip()
        if not zh or not en:
            empty += 1
        elif len(zh) > MAX_ZH_CHARS or len(en) > MAX_EN_CHARS:
            long_ += 1
        elif (zh, en) in seen:
            dup += 1
        else:
            seen.add((zh, en))
            cleaned.append((zh, en))
    print("\n语料清洗结果")
    print("-" * 40)
    print(f"原始：{len(pairs):,}  空：{empty:,}  过长：{long_:,}  重复：{dup:,}  清洗后：{len(cleaned):,}")
    return cleaned


def split_dataset(pairs):
    """按固定随机种子划分训练/验证/测试集。"""
    random.seed(RANDOM_SEED)
    pairs = random.sample(pairs, len(pairs))
    n = len(pairs)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)
    splits = (pairs[:train_end], pairs[train_end:val_end], pairs[val_end:])
    print("\n数据集划分结果")
    print("-" * 40)
    print(f"训练集：{len(splits[0]):,}  验证集：{len(splits[1]):,}  测试集：{len(splits[2]):,}")
    return splits


def save_pairs(pairs, output_path):
    """保存句对为 CSV。"""
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["zh", "en"])
        writer.writerows(pairs)
    print(f"已保存：{output_path}")


def main():
    if not ZH_PATH.exists():
        raise FileNotFoundError(f"找不到中文文件：{ZH_PATH}")
    if not EN_PATH.exists():
        raise FileNotFoundError(f"找不到英文文件：{EN_PATH}")

    pairs = clean_pairs(read_parallel_corpus(ZH_PATH, EN_PATH))
    if len(pairs) < 100:
        raise ValueError("清洗后的句对数量过少，请检查原始语料。")

    train_pairs, val_pairs, test_pairs = split_dataset(pairs)
    for name, data in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
        save_pairs(data, OUTPUT_DIR / f"{name}.csv")

    print("\n训练集样例")
    print("=" * 60)
    for zh, en in train_pairs[:5]:
        print(f"中文：{zh}\n英文：{en}\n" + "-" * 40)


if __name__ == "__main__":
    main()