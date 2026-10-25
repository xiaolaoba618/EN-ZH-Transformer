from pathlib import Path
import sentencepiece as spm
from config import *
BASE_DIR = Path(__file__).resolve().parents[2]

#INPUT_PATH = BASE_DIR / "data" / "tokenizer_train.txt"
#MODEL_DIR = BASE_DIR / 'tokenizer'
#MODEL_PREFIX = MODEL_DIR / 'opus_bpe'

# 训练语料
INPUT_PATH = "G:/Documents/Transformer_data/tokenizer_train.txt"

# 模型输出目录
MODEL_DIR = "G:/Documents"

# 输出模型前缀
MODEL_PREFIX = "G:/Documents/Transformer_data/opus_bpe"

ZH_PATH = BASE_DIR/"data"/"raw"/"MultiUN.en-zh.zh"
EN_PATH = BASE_DIR/"data"/"raw"/"MultiUN.en-zh.en"

# 输出目录
OUTPUT_DIR = BASE_DIR/"data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "tokenizer_train.txt"


def prepare_tokenizer_data():
    """Combine Chinese and English training sentences."""

    zh_lines = ZH_PATH.read_text(encoding="utf-8").splitlines()
    en_lines = EN_PATH.read_text(encoding="utf-8").splitlines()

    if len(zh_lines) != len(en_lines):
        raise ValueError(
            f"中英文行数不一致：ZH={len(zh_lines)}, EN={len(en_lines)}"
        )

    sentences = []

    for zh, en in zip(zh_lines, en_lines):
        zh = zh.strip()
        en = en.strip()

        # 删除空句子
        if not zh or not en:
            continue

        # 每个句子单独占一行
        sentences.append(zh)
        sentences.append(en)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(sentences) + "\n")

    print(f"原始句对数量：{len(zh_lines):,}")
    print(f"保留句对对应的句子数：{len(sentences):,}")
    print(f"输出路径：{OUTPUT_PATH}")



def train_tokenizer():
    """Train a shared Chinese-English BPE tokenizer."""

    spm.SentencePieceTrainer.train(
        input=INPUT_PATH,
        model_prefix=MODEL_PREFIX,

        # 使用 BPE 算法
        model_type="bpe",

        # 目标词表大小
        vocab_size=config['vocab_size'],

        # 特殊 Token 的 ID
        pad_id=config["pad_id"],

        unk_id=config["unk_id"],
        bos_id=config["bos_id"],
        eos_id=config["eos_id"],

        pad_piece="<PAD>",
        unk_piece="<UNK>",
        bos_piece="<BOS>",
        eos_piece="<EOS>",

        # 中文字符覆盖率
        character_coverage=0.9995,

    )

    print("SentencePiece 训练完成！")


if __name__ == "__main__":
    prepare_tokenizer_data()
    train_tokenizer()

