from config import config
import torch
import torch.nn as nn
import math


class PositionEncoding(nn.Module):
    def __init__(self, d_model=config["d_model"],head_dim = config['d_model'] // config['heads'],max_len=config["max_length"]):
        super(PositionEncoding, self).__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE要求head_dim必须为偶数")
        #单组二维向量旋转频率
        inv_freq = 1.0 / (config["pe_base"] ** (torch.arange(0,head_dim,2).float() / head_dim))
        #位置索引i
        positions = torch.arange(max_len).float()
        #每个位置旋转角度ceta
        angles = positions[:, None] * inv_freq[None, :]
        #正余弦编码并缓存
        self.register_buffer("sin", torch.sin(angles))
        self.register_buffer("cos", torch.cos(angles))

    def forward(self, x,start_pos = 0):
        #x[B,L,Dh]
        length = x.shape[1]
        if start_pos + length > self.sin.shape[0]:
            raise ValueError("序列超过MAX_LENGTHS")

        #计算当前Token对应位置角度
        sin = self.sin[start_pos:start_pos+length].unsqueeze(0) #[1,L,Dh/2]
        cos = self.cos[start_pos:start_pos+length].unsqueeze(0) #[1,L,Dh/2]

        #交错分组
        x_even = x[..., 0::2]  # 取偶数维[B,L,Dh/2]
        x_odd = x[..., 1::2]  # 取奇数维[B,L,Dh/2]

        #执行二维旋转
        rotated_even = x_even * cos - x_odd * sin #[B,L,Dh/2]
        rotated_odd = x_even * sin + x_odd * cos #[B,L,Dh/2]

        return torch.stack([rotated_even, rotated_odd], dim=-1).flatten(-2) #[B,L,Dh]

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size=config['vocab_size'], d_model=config["d_model"]):
        super(TokenEmbedding, self).__init__()
        #创建[vocab_size,D]的词向量表
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self,token_id):
        x = self.embedding(token_id) * math.sqrt(self.embedding.embedding_dim)
        return x

def attention(q, k, v, key_padding_mask=None, causal = False, query_start = 0,device = config['device']):
    #[B,Lq,Dh] @ [B,Dh,Lk] -> [B,Lq,Lk]
    scores = q @ k.transpose(-2,-1) / math.sqrt(q.shape[-1])
    #因果掩码
    if causal:
        q_pos = torch.arange(query_start,query_start+q.shape[-2],device=device)#生成对应q的一维向量
        k_pos = torch.arange(k.shape[-2],device=device)#生成对应k的一维向量
        mask = k_pos.unsqueeze(0) > q_pos.unsqueeze(1) #广播为条件指示函数 0 1 矩阵
        scores = scores.masked_fill_(mask.unsqueeze(0), float('-inf')) #把每一个掩码矩阵应用到batch的每一个样本上

    if key_padding_mask is not None:
        #[B,Lk] -> [B,1,Lk]
        #padding掩码
        scores = scores.masked_fill_(~key_padding_mask.unsqueeze(1), float('-inf'))

    weights = torch.softmax(scores, dim=-1)#[Lq,Lk]

    return weights @ v #[Lq,Lk] @ [Lk,Dh] -> [Lq,Dh]

class AttentionHead(nn.Module):
    def __init__(self, d_model=config['d_model'], heads=config['heads'],use_rope=True):
        super(AttentionHead, self).__init__()
        head_dim = d_model // heads
        #将qkv投影到单头注意力
        self.q = nn.Linear(d_model, head_dim, bias=False)
        self.k = nn.Linear(d_model, head_dim, bias=False)
        self.v = nn.Linear(d_model, head_dim, bias=False)
        #RoPE旋转编码注入
        self.rope = PositionEncoding(head_dim=head_dim, max_len=config['max_length']) if use_rope else None

    def forward(self,query,key,value,causal=False,key_padding_mask=None,cache=None,use_cache=None,static_kv=False,start_pos=0):
        #q只与当前样本有关，无需缓存,直接投影 [Lq,D] -> [Lq,Dh]
        q = self.q(query)
        # 交叉注意力，直接复用kv
        if static_kv and cache is not None:
            k,v = cache
        else:
            #自注意力部分，缓存KV
            k = self.k(key) #[Lq,D] -> [Lq,Dh]
            v = self.v(value)#[Lq,D] -> [Lq,Dh]
            # 增量自注意力
            q_start_pos = cache[0].shape[1] if cache is not None and not static_kv else start_pos
            k_start_pos = cache[0].shape[1] if cache is not None and not static_kv else start_pos
            if self.rope is not None:
                q = self.rope(q, start_pos=q_start_pos)
                k = self.rope(k, start_pos=k_start_pos)
            if cache is not None and not static_kv:
                k = torch.cat([cache[0],k],dim=1)
                v = torch.cat([cache[1],v],dim=1)
        #使用qk计算权重，汇总到v上，输出分数[Lq,Dh],query_start是已有缓存的token数，表示目前token的序列位置
        query_start = k.shape[1]-q.shape[1] if causal and use_cache else 0
        result = attention(q,k,v,causal=causal,key_padding_mask=key_padding_mask,query_start=query_start)
        return (result, (k, v)) if use_cache else result

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=config['d_model'], heads=config['heads'],use_rope=True):
        super(MultiHeadAttention, self).__init__()
        #每个注意力头独自学习QKV投影，参数独立，将特征拼接
        self.heads = nn.ModuleList(AttentionHead(heads=heads, d_model=d_model,use_rope=use_rope) for _ in range(heads))
        #将拼接后的特征投影，融合信息
        self.output_proj = nn.Linear(config['d_model'] , config['d_model'])

    def forward(self,query,key,value,causal=False,key_padding_mask=None,cache=None,use_cache = False,static_kv=False,start_pos=0):
        outputs = []
        new_cache = []
        for i, head in enumerate(self.heads):
            old = None if cache is None else cache[i]
            result = head(query,key,value,causal=causal,key_padding_mask=key_padding_mask,cache=old,use_cache=use_cache,static_kv=static_kv,start_pos=start_pos)
            if use_cache:
                head_output, head_cache = result
                outputs.append(head_output)
                new_cache.append(head_cache)
            else:
                outputs.append(result)
        joined = torch.cat(outputs, dim=-1)
        projected = self.output_proj(joined)
        if use_cache:
            return projected, new_cache
        return projected

class FeedForward(nn.Module):
    def __init__(self, d_model=config['d_model'], d_ff=config['d_ff'], dropout=config['dropout']):
        super(FeedForward, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.network(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model=config['d_model']):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention() #多头注意力
        self.ffn = FeedForward() #前馈神经网络
        self.norm1 = nn.LayerNorm(d_model)#自注意力残差连接层归一化
        self.norm2 = nn.LayerNorm(d_model)#FNN残差连接层归一化
    def forward(self, x, source_mask=None):
        #自注意力
        x = self.norm1(x + self.mha(x,x,x,key_padding_mask=source_mask))
        x = self.norm2(x + self.ffn(x))
        return x

class Encoder(nn.Module):
    def __init__(self,num_layers=config["num_layers"]):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(EncoderLayer() for _ in range(num_layers))

    def forward(self,x,source_mask=None):
        for layer in self.layers:
            x = layer(x,source_mask=source_mask)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model=config['d_model'], d_ff=config['d_ff'], dropout=config['dropout']):
        super(DecoderLayer, self).__init__()
        self.self_mha = MultiHeadAttention()
        self.cross_mha = MultiHeadAttention(use_rope=False)
        self.ffn = FeedForward()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, source_mask=None, target_mask=None, cache=None, use_cache=False, start_pos=0):
        if use_cache:
            # 掩码自注意力
            old_self = None if cache is None else cache["self"]
            old_cross = None if cache is None else cache["cross"]
            self_out, new_self = self.self_mha(x, x, x, causal=True, cache=old_self, use_cache=True,start_pos=start_pos)

            x = self.norm1(x + self_out)
            # 交叉注意力
            cross_out, new_cross = self.cross_mha(x, encoder_output, encoder_output, key_padding_mask=source_mask,cache=old_cross, use_cache=True, static_kv=True, start_pos=start_pos)
            x = self.norm2(x + cross_out)
            # FFN
            x = self.norm3(x + self.ffn(x))
            return x, {"self": new_self, "cross": new_cross}
        # 训练使用完整序列和掩码，不缓存
        x = self.norm1(x + self.self_mha(x, x, x, causal=True, key_padding_mask=target_mask))
        # 交叉注意力，k,v 来自编码器，q 来自已生成的上文
        x = self.norm2(x + self.cross_mha(x, encoder_output, encoder_output, key_padding_mask=source_mask))
        # FFN
        x = self.norm3(x + self.ffn(x))
        return x  # 最终输出

class Decoder(nn.Module):
    def __init__(self, num_layers=config["num_layers"]):
        super().__init__()
        self.layers = nn.ModuleList(DecoderLayer() for _ in range(num_layers))

    def forward(self, x, encoder_output,source_mask=None,target_mask=None,caches=None,use_cache=False,start_pos=0):
        new_caches = []
        for i, layer in enumerate(self.layers):
            # 当前层对应的历史缓存
            old_cache = None if caches is None else caches[i]
            if use_cache:
                # 增量解码：当前层返回输出和更新后的缓存
                x, layer_cache = layer(x, encoder_output, source_mask=source_mask, target_mask=target_mask, cache=old_cache, use_cache=True,start_pos=start_pos)
                new_caches.append(layer_cache)
            else:
                # 训练：完整目标序列，不使用 KV Cache
                x = layer(x, encoder_output, source_mask=source_mask, target_mask=target_mask, cache=None, use_cache=False)
        if use_cache:
            return x, new_caches
        return x

class Transformer(nn.Module):
    def __init__(self,num_layers=config["num_layers"],d_model=config['d_model']):
        super(Transformer, self).__init__()
        vocab_size = config['vocab_size']
        #tokenembedding完成查词向量 -> 乘以sqrt(D) ->位置编码
        self.source_embedding = TokenEmbedding(vocab_size)
        self.target_embedding = TokenEmbedding(vocab_size)

        self.encoder = Encoder(num_layers)
        self.decoder = Decoder(num_layers)
        #每个输出位置的向量表示映射为token分数
        self.output_projection = nn.Linear(d_model, vocab_size)
        #初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)

        elif isinstance(module, nn.LayerNorm):
            if module.elementwise_affine:
                nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(self,source):
        #输入语句
        #padding掩码
        source_mask=source.ne(config['pad_id'])
        return self.encoder(self.source_embedding(source),source_mask=source_mask)

    def decode(self,target,encoder_output,source_mask=None):
        x = self.target_embedding(target)
        #padding掩码
        target_mask = target.ne(config['pad_id'])
        x = self.decoder(x,encoder_output,source_mask=source_mask,target_mask=target_mask)
        #输出logits
        return self.output_projection(x)

    def decode_step(self,new_token,encoder_output,source_mask,caches=None):
        start_pos = 0 if caches is None else caches[0]['self'][0][0].shape[1]
        x = self.target_embedding(new_token)
        x,new_caches = self.decoder(x,encoder_output,source_mask=source_mask,caches=caches,use_cache=True,start_pos=start_pos)
        return self.output_projection(x),new_caches

    def forward(self,source,target):
        encoder_output = self.encode(source)
        return self.decode(target,encoder_output,source_mask=source.ne(config['pad_id']))






















