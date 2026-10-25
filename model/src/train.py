from pathlib import Path
import sys
import math
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import ctypes

# Windows 防休眠
def prevent_sleep():
    # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000002)

def allow_sleep():
    ctypes.windll.kernel32.SetThreadExecutionState(0x00000000)



# 路径与导入
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(1, str(ROOT_DIR))

from config import config
from model import Transformer
from utils import TranslationDataset, collate_fn, sp, DATA_DIR

MODEL_DIR = ROOT_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_PATH = MODEL_DIR / "best_transformer.pth"
LAST_MODEL_PATH = MODEL_DIR / "last_transformer.pth"

def cfg_get(*keys, default=None):
    """按顺序查找 config 键，兼容大小写写法（如 RESUME / resume）。"""
    for k in keys:
        if k in config:
            return config[k]
    return default


# 训练参数
SEED = int(cfg_get("seed", default=42))
EPOCHS = int(cfg_get("max_epochs", default=20))
BATCH_SIZE = int(cfg_get("batch_size", default=16))
NUM_WORKERS = int(cfg_get("num_workers", default=0))
LEARNING_RATE = float(cfg_get("learning_rate", "max_lr", default=3e-4))
WEIGHT_DECAY = float(cfg_get("weight_decay", default=1e-2))
# 优先用显式的 warmup_steps；config 没给才退回按 warmup_ratio 比例算
WARMUP_STEPS = int(cfg_get("warmup_steps")) if cfg_get("warmup_steps") is not None else None
WARMUP_RATIO = float(cfg_get("warmup_ratio", default=0.05))
GRAD_CLIP = float(cfg_get("grad_clip", "GRAD_CLIP", default=1.0))
PATIENCE = int(cfg_get("patience", default=5))
RESUME = bool(cfg_get("RESUME", "resume", default=True))     # 默认自动续训
SAVE_EVERY = int(cfg_get("save_every_steps", default=2000))  # 每多少步存一次检查点

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(config["device"])
print(f"训练设备：{device}  Batch size：{BATCH_SIZE}  Epochs：{EPOCHS}  Workers：{NUM_WORKERS}")

_WARMUP_DESC = (f"固定 {WARMUP_STEPS:,} 步" if WARMUP_STEPS is not None
                else f"按比例 {WARMUP_RATIO}（=总步数×{WARMUP_RATIO}）")
print(f"学习率 {LEARNING_RATE:.2e}  |  梯度裁剪 {GRAD_CLIP}  |  每 {SAVE_EVERY} 步存盘  |  "
      f"断点续训 {'开' if RESUME else '关'}  |  warmup：{_WARMUP_DESC}")

if cfg_get("learning_rate") is None:
    print(f"⚠️ config.py 未指定 learning_rate，当前取 {LEARNING_RATE:.2e}")
if WARMUP_STEPS is None:
    print(f"⚠️ config.py 未指定 warmup_steps，改用 warmup_ratio 比例计算")


def main():
    # 加载数据并划分训练/验证集
    dataset = TranslationDataset(
        csv_path=DATA_DIR / "train.csv",
        tokenizer=sp,
        max_src_len=config["max_length"],
        max_tgt_len=config["max_length"],
    )
    if len(dataset) < 2:
        raise ValueError("数据集不足两条，无法划分训练集和验证集。")

    train_size = int(len(dataset) * 0.9)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED)
    )

    # persistent_workers：worker 不随 epoch 重建，省掉每个 epoch 重新传送数据集的开销
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )
    print(f"总样本：{len(dataset):,}  训练：{len(train_dataset):,}  验证：{len(val_dataset):,}  批次数：{len(train_loader)}")

    # 模型、损失、优化器
    model = Transformer().to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=config["pad_id"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Warmup + Cosine 衰减
    total_steps = max(1, len(train_loader) * EPOCHS)
    warmup_steps = WARMUP_STEPS if WARMUP_STEPS is not None else int(total_steps * WARMUP_RATIO)
    print(f"总步数：{total_steps:,}   warmup：{warmup_steps:,} 步"
          f"（约 {warmup_steps / max(1, len(train_loader)):.2f} 个 epoch）")

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-3, (step + 1) / warmup_steps)
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # 断点续训
    start_epoch = 0
    resume_step = 0
    best_val_loss = float("inf")
    bad_epochs = 0

    def save_checkpoint(epoch, step, note=""):
        torch.save({
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "bad_epochs": bad_epochs,
            "config": config,
        }, LAST_MODEL_PATH)
        if note:
            print(f"  💾 检查点已保存（{note}）：{LAST_MODEL_PATH.name}")

    if RESUME and LAST_MODEL_PATH.exists():
        try:
            ckpt = torch.load(LAST_MODEL_PATH, map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            best_val_loss = ckpt["best_val_loss"]
            bad_epochs = ckpt.get("bad_epochs", 0)
            if ckpt.get("step") is None:
                # 旧格式：只在 epoch 结束时保存过
                start_epoch, resume_step = ckpt["epoch"] + 1, 0
            else:
                # 新格式：记录到 epoch 内的第几步
                start_epoch, resume_step = ckpt["epoch"], int(ckpt["step"])
            if resume_step >= len(train_loader):
                start_epoch, resume_step = start_epoch + 1, 0
            print(f"✅ 已从检查点恢复：将训练 Epoch {start_epoch + 1}，"
                  f"本 epoch 已完成 {resume_step:,}/{len(train_loader):,} 步，"
                  f"最佳验证 Loss {best_val_loss:.4f}")
        except Exception as exc:
            print(f"⚠️ 检查点加载失败（{exc}），改为从头训练。"
                  f"若想强制从头，请删除 {LAST_MODEL_PATH}")
            start_epoch, resume_step, best_val_loss, bad_epochs = 0, 0, float("inf"), 0
    elif RESUME:
        print("未发现检查点，从头开始训练。")
    else:
        print("resume=False，本次从头开始训练。")

    def train_one_epoch(epoch, skip_steps=0):
        model.train()
        total_loss, total_tokens = 0.0, 0
        stopped_at = 0
        interrupted = False
        if skip_steps > 0:
            print(f"  续训：跳过本 epoch 前 {skip_steps:,} 步...")
        for step, batch in enumerate(train_loader):
            if step < skip_steps:
                if (step + 1) % 20000 == 0:
                    print(f"    跳过中 {step + 1:,}/{skip_steps:,}")
                continue
            try:
                src = batch["src"].to(device, non_blocking=True)
                tgt_input = batch["tgt_input"].to(device, non_blocking=True)
                tgt_output = batch["tgt_output"].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = model(src, tgt_input)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))

                if not torch.isfinite(loss):
                    raise RuntimeError(f"Loss 出现 NaN/Inf，Epoch 训练步：{step}")

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
                optimizer.step()
                scheduler.step()

                valid_tokens = tgt_output.ne(config["pad_id"]).sum().item()
                total_loss += loss.item() * valid_tokens
                total_tokens += valid_tokens

                if (step + 1) % 100 == 0 or step == 0:
                    print(f"  Step [{step + 1}/{len(train_loader)}] Loss={loss.item():.4f} "
                          f"LR={optimizer.param_groups[0]['lr']:.2e}")

                # 关键：epoch 内定期落盘，Ctrl+C 最多丢 SAVE_EVERY 步
                if (step + 1) % SAVE_EVERY == 0:
                    save_checkpoint(epoch, step + 1)

                del logits, loss
            except KeyboardInterrupt:
                interrupted = True
                break
            stopped_at = step + 1
        return total_loss / max(1, total_tokens), stopped_at, interrupted

    @torch.no_grad()
    def validate():
        model.eval()
        total_loss, total_tokens = 0.0, 0
        for batch in val_loader:
            src = batch["src"].to(device, non_blocking=True)
            tgt_input = batch["tgt_input"].to(device, non_blocking=True)
            tgt_output = batch["tgt_output"].to(device, non_blocking=True)

            logits = model(src, tgt_input)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError("验证损失出现 NaN/Inf。")

            valid_tokens = tgt_output.ne(config["pad_id"]).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens
            del logits, loss
        return total_loss / max(1, total_tokens)

    # 训练循环 + 最佳模型/检查点保存 + 早停
    try:
        for epoch in range(start_epoch, EPOCHS):
            print(f"\n{'=' * 60}\nEpoch {epoch + 1}/{EPOCHS}\n{'=' * 60}")

            skip = resume_step if epoch == start_epoch else 0
            train_loss, stopped_at, interrupted = train_one_epoch(epoch, skip)

            if interrupted:
                save_checkpoint(epoch, stopped_at, note=f"中断于 Epoch {epoch + 1} 第 {stopped_at:,} 步")
                print(f"\n检测到手动中断。已保存到 Epoch {epoch + 1} 第 {stopped_at:,} 步，"
                      f"下次运行会自动从这里继续。")
                break

            val_loss = validate()
            print(f"Train Loss：{train_loss:.4f}  Val Loss：{val_loss:.4f}  LR：{optimizer.param_groups[0]['lr']:.2e}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                bad_epochs = 0
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(f"✅ 最佳模型已保存：{BEST_MODEL_PATH}")
            else:
                bad_epochs += 1
                print(f"验证集未改善：{bad_epochs}/{PATIENCE}")

            save_checkpoint(epoch, len(train_loader), note=f"Epoch {epoch + 1} 结束")

            if bad_epochs >= PATIENCE:
                print("触发 Early Stopping，结束训练。")
                break

        else:
            print(f"\n训练结束。最佳验证集 Loss：{best_val_loss:.4f}\n最佳模型：{BEST_MODEL_PATH}")

    except KeyboardInterrupt:
        print("\n检测到手动中断（验证阶段）。最近一次保存的检查点可用于恢复训练。")

if __name__ == "__main__":
    prevent_sleep()
    print("✅ 已开启防休眠，脚本运行时电脑不会睡眠")
    try:
        main()
    finally:
        allow_sleep()
        print("✅ 恢复休眠策略")
