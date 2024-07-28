import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        '''
        for m in self.modules():
            print(m)
            m.register_backward_hook(self.get_attention_gradient)
        '''
        #exit()
        #self.attention_gradients = []

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
    ''' 
    def get_attention_gradient(self, module, grad_input, grad_output):
        #self.attention_gradients.append(grad_input[0].cpu())
        print(grad_input,grad_input[0].shape)
    '''
    
        
        

    def forward(self, values, keys, query, mask):
        # Get number of training examples
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, heads, head_dim)
        queries = self.queries(query)  # (N, query_len, heads, heads_dim)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)

        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        #attention.retain_grad()
        #print(attention)
        #exit()
        #attention.register_backward_hook(self.get_attn_gradients)
        #print('AG',self.attention_gradients)
        #exit()
        
        #print(attention.shape,attention[0][1].shape)
        #exit()
        # attention shape: (N, heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)

        return out,attention

class TransformerBlock(nn.Module):
    
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention, attention_softmax = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out,attention_softmax

class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):

        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        #self.position_embedding = nn.Embedding(max_length, embed_size)

        # self.layers = TransformerBlock(
        #             embed_size,
        #             heads,
        #             dropout=dropout,
        #             forward_expansion=forward_expansion,
        #         )

        self.layers = nn.ModuleList(
                    [
                        TransformerBlock(
                            embed_size,
                            heads,
                            dropout=dropout,
                            forward_expansion=forward_expansion,
                        )
                        for _ in range(num_layers)
                    ]
                )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(
            (self.word_embedding(x))
        )

        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case.
        # for layer in self.layers:
        # out = self.layers(out, out, out, mask)
        for layer in self.layers:
            out, attention_softmax = layer(out, out, out, mask)

        return out,attention_softmax

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size_1,
        src_vocab_size_2,
        src_vocab_size_3,
        src_pad_idx,
        embed_size=512,
        num_layers=1,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cpu",
        max_length_1=512,
        max_length_2=512,
        max_length_3=512,
    ):

        super(Transformer, self).__init__()

        self.encoder1 = Encoder(
            src_vocab_size_1,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length_1,
        )
        self.encoder2 = Encoder(
            src_vocab_size_2,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length_2,
        )
        self.encoder3 = Encoder(
            src_vocab_size_3,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length_3,
        )
        

        self.src_pad_idx = src_pad_idx
        self.device = device
        self.fc1 = nn.Linear((max_length_1+max_length_2+max_length_3)*embed_size, 64)
        self.out = nn.Linear(64, 1)
        self.bn = nn.BatchNorm1d(64)

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)



    def forward(self, src_1,src_2,src_3):
        src_mask_1 = self.make_src_mask(src_1)
        enc_src_1,as1 = self.encoder1(src_1, src_mask_1)
        enc_src_1 = enc_src_1.reshape(enc_src_1.shape[0], -1)

        src_mask_2 = self.make_src_mask(src_2)
        enc_src_2,as2 = self.encoder2(src_2, src_mask_2)
        enc_src_2 = enc_src_2.reshape(enc_src_2.shape[0], -1)

        src_mask_3 = self.make_src_mask(src_3)
        enc_src_3,as3 = self.encoder3(src_3, src_mask_3)
        enc_src_3 = enc_src_3.reshape(enc_src_3.shape[0], -1)

        enc_src=torch.cat([enc_src_1, enc_src_2, enc_src_3], dim=1)
        #print(enc_src_1.shape,enc_src_2.shape,enc_src_3.shape)
        #print(enc_src.shape)
        #exit()

        x = self.bn(self.fc1(enc_src))
        x = self.out(x)

        return x,as1,as2,as3







