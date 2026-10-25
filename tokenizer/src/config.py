import random
import torch
import sentencepiece as spm


sp = spm.SentencePieceProcessor(r"G:\Documents\Transformer_data\opus_bpe.model")
#sp = spm.SentencePieceProcessor( BASE_DIR / 'tokenizer' / 'opus_bpe.model')
config = {
    'd_model' : 128, #单个Token向量维数
    'heads' : 4 ,   #注意力头数
    'd_ff' : 512,  #FFN隐藏层维数
    'dropout' : 0.1,#随机丢弃比例
    'num_layers' : 6,#Encoder/Decoder堆叠层数
    'max_length' : 128,#输入输出Token最大长度
    'batch_size' : 16,#单批训练句对数
    'max_lr' : 5e-4,  #最大学习率
    'min_lr' : 1e-5,  #最小学习率
    'weight_decay' : 1e-4,#权重衰减参数
    'max_epochs' : 100,#最大训练轮数
    'patience' : 10,#早停阈值
    'warmup_epochs' : 3,#预热轮数
    'device' : 'cuda:0',#训练设备
    'seed' : 42,#随机种子
    'pad_id' : sp.pad_id(),#<PAD>Token_ID
    'unk_id' : sp.unk_id(),#<UNK>Token_ID
    'bos_id' : sp.bos_id(),#<BOS>Token_ID
    'eos_id' : sp.eos_id(),#<EOS>Token_ID
    'num_workers' : 0,#训练设备数
    'vocab_size' : sp.get_piece_size(),#词表数量（中英共享一张词表）
    'pe_base' : 10000.0,#旋转位置编码分母超参
}
