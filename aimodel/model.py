import math
import torch
import torch.nn as nn
class SelfAttention(nn.Module):
    def __init__(self, emb_size, heads):
        super(SelfAttention, self).__init__()
        self.emb_size = emb_size
        self.heads = heads
        self.head_dim = emb_size // heads

        assert (self.head_dim * heads == emb_size), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, emb_size)

    def forward(self, values, keys, query, mask):
        batch_size = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(batch_size, value_len, self.heads, self.head_dim)
        keys = keys.reshape(batch_size, key_len, self.heads, self.head_dim)
        queries = query.reshape(batch_size, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Calculate energy
        # energy = torch.matmul(queries, keys.transpose(1, 2)) # (batch_size, head, query_len, key_len)')
        energy = torch.einsum('bqhd,bkhd->bhqk', [queries, keys]) # (batch_size, head, query_len, key_len)')

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        d_k = keys.shape[3]
        attention = torch.softmax(energy / (d_k ** 0.5), dim=3) # (batch_size, head, query_len, key_len)

        out = torch.einsum('bhqk,bvhd->bqhd', [attention, values]) # (batch_size, query_len, head, embed_dim)
        out = out.reshape(batch_size, query_len, self.heads * self.head_dim) # (batch_size, query_len, embed_dim)
        out = self.fc_out(out) # (batch_size, query_len, embed_dim)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, emb_size, heads, forward_expansion, drop_out) -> None:
        super(TransformerBlock, self).__init__()

        self.attention = SelfAttention(emb_size, heads)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(in_features=emb_size, out_features=forward_expansion * emb_size, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=forward_expansion * emb_size, out_features=emb_size, bias=True),
        )

        self.dropout = nn.Dropout(drop_out)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        norm = self.norm1(attention + query)
        norm = self.dropout(norm)

        forward = self.feed_forward(norm)
        out = self.norm2(forward + norm)
        return out

class WordPositionEmbedding(nn.Module):
    def __init__(self, vocab_size, max_seq_len, emb_size, device, fixed=True):
        super(WordPositionEmbedding, self).__init__()

        self.device = device
        self.fixed = fixed

        self.word_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_size, device=device)

        if fixed:
            # Create fixed (non-learnable) position embeddings
            position = torch.arange(max_seq_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / emb_size))
            position_embedding = torch.zeros(max_seq_len, emb_size)
            position_embedding[:, 0::2] = torch.sin(position * div_term)
            position_embedding[:, 1::2] = torch.cos(position * div_term)
            # Register position_embedding as a buffer
            self.register_buffer('position_embedding', position_embedding)
        else:
            self.position_embedding = nn.Embedding(max_seq_len, emb_size)

    def forward(self, X):
        batch_size, seq_len = X.shape

        # Get word embeddings
        word = self.word_embedding(X)

        if self.fixed:
            # Use fixed position embeddings
            position = self.position_embedding[:seq_len, :]
        else:
            # Get position embeddings
            position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, seq_len)
            position = self.position_embedding(position_ids)

        # Add word and position embeddings
        embeddings = word + position
        return embeddings

class Encoder(nn.Module):
    def __init__(self, vocab_size, seq_len, emb_size, n_layers, heads, forward_expansion, drop_out, device):
        super(Encoder, self).__init__()

        self.emb_size = emb_size
        self.device = device

        self.embedding = WordPositionEmbedding(vocab_size, seq_len, emb_size, device, fixed=True)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(emb_size, heads, forward_expansion, drop_out) for _ in range(n_layers)
            ]
        )

        self.dropout = nn.Dropout(drop_out)

    def forward(self, X, mask):
        batch_size, seq_len = X.shape
        out = self.dropout(self.embedding(X))

        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out

class DecoderBlock(nn.Module):
    def __init__(self, emb_size, heads, forward_expansion, drop_out) -> None:
        super(DecoderBlock, self).__init__()

        self.attention = SelfAttention(emb_size, heads)
        self.norm = nn.LayerNorm(emb_size)
        self.transformer_block = TransformerBlock(emb_size, heads, forward_expansion, drop_out)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, X, value, key, src_mask, trg_mask):
        attention = self.attention(X, X, X, trg_mask)
        query = self.dropout(self.norm(attention + X))
        out = self.transformer_block(value, key, query, src_mask)
        return out
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, seq_len, emb_size, n_layers, heads, forward_expansion, drop_out, device) -> None:
        super(Decoder, self).__init__()

        self.device = device
        self.embedding = WordPositionEmbedding(vocab_size, seq_len, emb_size, device, fixed=True)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(emb_size, heads, forward_expansion, drop_out) for _ in range(n_layers)
            ]
        )

        self.fc_out = nn.Linear(emb_size, vocab_size)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, X, enc_out, src_mask, trg_mask):
        batch_size, seq_len = X.shape

        out = self.dropout(self.embedding(X))

        for layer in self.layers:
            out = layer(out, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(out)
        return out
class TransformerPytroch(nn.Module):
    def __init__(
        self,
        inp_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        emb_size,
        n_layers=1,
        heads=1,
        forward_expansion=1,
        drop_out=0.2,
        max_seq_len=100,
        device=torch.device('cuda')
    ):
        super(TransformerPytroch, self).__init__()

        self.enc_embedding = WordPositionEmbedding(inp_vocab_size, max_seq_len, emb_size, device, fixed=False)
        self.dec_embedding = WordPositionEmbedding(trg_vocab_size, max_seq_len, emb_size, device, fixed=False)


        self.device = device
        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=forward_expansion,
            dropout=drop_out,
            batch_first=True,
            device=device
        )
        self.fc_out = nn.Linear(emb_size, trg_vocab_size)
        self.dropout = nn.Dropout(drop_out)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_src_mask(self, src):
        src_mask = src == self.src_pad_idx

        # (N, src_len)
        return src_mask.to(self.device)

    def forward(self, src, trg):
        batch_size, trg_seq_length = trg.shape

        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(
            self.device
        )

        src_emb = self.dropout(self.enc_embedding(src))
        trg_emb = self.dropout(self.dec_embedding(trg))

        out = self.transformer(
            src_emb,
            trg_emb,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask,
        )
        out = self.fc_out(out)
        return out