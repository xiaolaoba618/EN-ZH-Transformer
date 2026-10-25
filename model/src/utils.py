from pathlib import Path
import csv
import sys
import random
from array import array

import numpy as np
import torch
import sentencepiece as spm
from torch.utils.data import Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence

# 路径配置
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
TOKENIZER_PATH = Path(r"G:\Documents\Transformer_data\opus_bpe.model")
CACHE_DIR = BASE_DIR / "cache"  # 缓存文件夹
CACHE_DIR.mkdir(parents=True, exist_ok=True)
sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_PATH))
PAD_ID, UNK_ID, BOS_ID, EOS_ID = sp.pad_id(), sp.unk_id(), sp.bos_id(), sp.eos_id()
print(f"词表大小：{sp.get_piece_size()}  PAD={PAD_ID} UNK={UNK_ID} BOS={BOS_ID} EOS={EOS_ID}")

# csv 模块默认单字段上限 131072 字符，语料里的长句会直接抛 _csv.Error
try:
    csv.field_size_limit(2 ** 31 - 1)
except (OverflowError, ValueError):
    csv.field_size_limit(sys.maxsize)


class TranslationDataset(Dataset):
    """中英翻译数据集，带 token 结果缓存。

    存储方式：把全部 token 压平进两个 int32 大张量（src / tgt），另用 offsets
    数组记录每条样本的边界，__getitem__ 时按 offsets 切片还原。

    旧版把每条样本存成 3 个独立 tensor 对象，实测每样本 2425 字节，而真实数据
    只有 600 字节，其余全是 torch 对象头开销 —— 6.9M 样本约 18GB，必然 OOM。
    压平后同样数据约 1.8GB。

    对外接口与旧版完全一致：
      - __len__() -> int
      - __getitem__(idx) -> {"src", "tgt_input", "tgt_output"}，均为 torch.long 一维张量
    """

    def __init__(self, csv_path, tokenizer, max_src_len=128, max_tgt_len=128):
        self.tokenizer = tokenizer
        self.pad_id, self.bos_id, self.eos_id = tokenizer.pad_id(), tokenizer.bos_id(), tokenizer.eos_id()
        self.max_src_len, self.max_tgt_len = max_src_len, max_tgt_len

        csv_path = Path(csv_path)
        # 缓存 key 必须包含所有会影响结果的变量，否则改了 config 会静默加载旧缓存
        cache_path = CACHE_DIR / (
            f"{csv_path.stem}_s{max_src_len}_t{max_tgt_len}"
            f"_v{tokenizer.get_piece_size()}_{csv_path.stat().st_size}.pt"
        )

        loaded = None
        if cache_path.exists():
            try:
                loaded = torch.load(cache_path, weights_only=True)
            except Exception as exc:
                print(f"⚠️ 缓存损坏，将重新生成：{exc}")
                loaded = None
            if loaded is not None and not self._cache_valid(loaded):
                print("⚠️ 缓存与当前配置不一致，将重新生成。")
                loaded = None

        if loaded is not None:
            print(f"✅ 加载缓存：{cache_path}")
            self.src_data = loaded["src_data"]
            self.tgt_data = loaded["tgt_data"]
            self.src_off = loaded["src_off"].numpy()
            self.tgt_off = loaded["tgt_off"].numpy()
            self._n = len(self.src_off) - 1
        else:
            print(f"⏳ 第一次处理数据集，生成缓存：{csv_path}")
            self._build(csv_path)
            torch.save({
                "src_data": self.src_data,
                "tgt_data": self.tgt_data,
                "src_off": torch.from_numpy(self.src_off),
                "tgt_off": torch.from_numpy(self.tgt_off),
                "num_samples": self._n,
                "max_src_len": max_src_len,
                "max_tgt_len": max_tgt_len,
                "vocab_size": tokenizer.get_piece_size(),
            }, cache_path)
            print(f"✅ 缓存保存完成：{cache_path}")

        print(f"样本数：{self._n:,}   src token：{len(self.src_data):,}   tgt token：{len(self.tgt_data):,}")

    @staticmethod
    def _cache_valid(loaded):
        need = {"src_data", "tgt_data", "src_off", "tgt_off", "num_samples"}
        if not need.issubset(loaded.keys()):
            return False
        if len(loaded["src_off"]) != len(loaded["tgt_off"]):
            return False
        if loaded["num_samples"] != len(loaded["src_off"]) - 1:
            return False
        return True

    def _build(self, csv_path):
        tokenizer = self.tokenizer
        bos_id, eos_id = self.bos_id, self.eos_id
        src_cap, tgt_cap = self.max_src_len - 1, self.max_tgt_len - 1
        encode = tokenizer.encode

        # array 按原始字节存 int32/int64，不产生 Python int 对象，避免中间态爆内存
        src_flat, tgt_flat = array("i"), array("i")
        src_off, tgt_off = array("q", [0]), array("q", [0])
        n = 0

        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                raise ValueError(f"{csv_path} 是空文件。")
            cols = [h.strip().lower() for h in header]
            try:
                i_zh, i_en = cols.index("zh"), cols.index("en")
            except ValueError:
                i_zh, i_en = 0, 1
            width = max(i_zh, i_en) + 1

            for row in reader:
                if len(row) < width:
                    continue
                zh, en = row[i_zh].strip(), row[i_en].strip()
                if not zh or not en:
                    continue
                src_flat.extend(encode(zh, out_type=int)[:src_cap])
                src_flat.append(eos_id)
                src_off.append(len(src_flat))
                tgt_flat.append(bos_id)
                tgt_flat.extend(encode(en, out_type=int)[:tgt_cap])
                tgt_flat.append(eos_id)
                tgt_off.append(len(tgt_flat))
                n += 1
                if n % 1_000_000 == 0:
                    print(f"  已编码 {n:,} 条...")

        if n == 0:
            raise ValueError(f"{csv_path} 中没有有效的中英句对。")

        # 零拷贝转成 torch 张量；_keepalive 保证底层 array buffer 不被回收
        self.src_data = torch.from_numpy(np.frombuffer(src_flat, dtype=np.int32))
        self.tgt_data = torch.from_numpy(np.frombuffer(tgt_flat, dtype=np.int32))
        # offsets 只有几十 MB，直接拷一份，彻底摆脱生命周期问题
        self.src_off = np.frombuffer(src_off, dtype=np.int64).copy()
        self.tgt_off = np.frombuffer(tgt_off, dtype=np.int64).copy()
        self._keepalive = (src_flat, tgt_flat)
        self._n = n

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        s0, s1 = self.src_off[idx], self.src_off[idx + 1]
        t0, t1 = self.tgt_off[idx], self.tgt_off[idx + 1]
        src = self.src_data[s0:s1].long()
        tgt = self.tgt_data[t0:t1].long()
        return {"src": src, "tgt_input": tgt[:-1], "tgt_output": tgt[1:]}

    # num_workers>0 时，DataLoader 会把整个数据集 pickle 给每个 worker 进程。
    # _keepalive 里的 array.array 只能按值序列化，等于把全部 token 数据再复制一遍
    # （实测 pickle 体积翻倍，6.9M 数据 × 4 worker ≈ 多占 14GB）。
    # worker 反序列化后拿到的是独立的 torch 张量，根本不需要它，故在 pickle 时剔除。
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_keepalive", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._keepalive = ()

    @property
    def lengths(self):
        """每条样本的 tgt 长度（numpy 数组），供长度分桶采样使用。"""
        return self.tgt_off[1:] - self.tgt_off[:-1]


def collate_fn(batch):
    def pad(key):
        return pad_sequence([item[key] for item in batch], batch_first=True, padding_value=PAD_ID)
    return {"src": pad("src"), "tgt_input": pad("tgt_input"), "tgt_output": pad("tgt_output")}


class LengthGroupedSampler(Sampler):
    """长度分桶采样器（可选，不影响默认行为）。

    把长度相近的句子聚到同一个 batch，padding 浪费能从约 47% 降到 10% 以内。
    用法（注意：传了 sampler 就不要再传 shuffle=True，否则报错）：

        loader = DataLoader(train_dataset, batch_size=128,
                            sampler=LengthGroupedSampler(train_dataset.lengths, 128),
                            collate_fn=collate_fn)

    若配合 random_split 使用，采样器下标是「子集内的下标」，必须同步切片：

        train_ds, val_ds = random_split(dataset, [n_train, n_val])
        lengths = dataset.lengths[train_ds.indices]
        sampler = LengthGroupedSampler(lengths, 128)
    """

    def __init__(self, lengths, batch_size, shuffle=True, seed=42, mega_factor=50):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.mega_factor = mega_factor
        self.epoch = 0

    def __len__(self):
        return len(self.lengths)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        idx = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(idx)

        mega = self.batch_size * self.mega_factor
        batches = []
        for start in range(0, len(idx), mega):
            chunk = idx[start:start + mega]
            chunk.sort(key=lambda i: self.lengths[i])
            for j in range(0, len(chunk), self.batch_size):
                batches.append(chunk[j:j + self.batch_size])
        if self.shuffle:
            rng.shuffle(batches)
        return iter([i for b in batches for i in b])
