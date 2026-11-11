import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len):
        super(DecoderOnlyTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embed_dim, max_seq_len * 2)

        decoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        b, seq_len = x.size()
        device = x.device

        mask = self.generate_square_subsequent_mask(seq_len).to(device)

        emb = self.embedding(x)
        emb = self.pos_encoder(emb)

        output = self.transformer(emb, mask=mask, is_causal=True)
        logits = self.fc_out(output)
        return logits

class KVBaseline(nn.Module):
    def __init__(self, vocab_size, embed_dim, memory_size, max_seq_len):
        super(KVBaseline, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.memory_size = memory_size

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len * 2, embed_dim)

        self.fc_q = nn.Linear(embed_dim, memory_size)
        self.fc_k = nn.Linear(embed_dim, memory_size)
        self.fc_v = nn.Linear(embed_dim, memory_size)

        self.fc_out = nn.Linear(embed_dim + memory_size, vocab_size)

    def forward(self, x):
        b, seq_len = x.size()
        device = x.device

        pos = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0).expand(b, seq_len).to(device)
        emb = self.embedding(x) + self.pos_embedding(pos)

        out_logits = []

        for t in range(seq_len):
            xt = emb[:, t, :]
            q_t = self.fc_q(xt)

            if t == 0:
                read_val = torch.zeros(b, self.memory_size).to(device)
            else:
                k_past = self.fc_k(emb[:, :t, :])
                v_past = self.fc_v(emb[:, :t, :])
                scores = torch.bmm(k_past, q_t.unsqueeze(-1)).squeeze(-1)
                attn = F.softmax(scores, dim=-1)
                read_val = torch.bmm(attn.unsqueeze(1), v_past).squeeze(1)

            combined = torch.cat([xt, read_val], dim=-1)
            logits = self.fc_out(combined)
            out_logits.append(logits.unsqueeze(1))

        out_logits = torch.cat(out_logits, dim=1)
        return out_logits
