import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embd: int, n_tokens: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        self.position_embedding = nn.Parameter(torch.zeros(n_token, n_embd))

    def forward(self, tokens):
        #(batch_size, Sequence_Length) -> (batch_size, Sequence_Length, Embedding_Dimension)
        x = self.token_embedding(tokens)

        x += self.position_embedding

        return x
    

class CLIPLayer(nn.Module):

    def __init__(self, n_heads: int, n_embed: int):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(n_embed)
        self.attention = SelfAttention(n_heads, n_embed)
        self.layernorm_2 = nn.LayerNorm(n_embed)

        self.linear_1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear_2 == nn.Linear(4 * n_embed, n_embed)

    def forward(self,x : torch.Tensor) -> torch.Tensor:
        #(batch_size, Sequence_Length, Embedding_Dimension)

        residue = x

        ##SELF attention

        x.self.layernorm_1(x)

        x.self.attention(x, causal_mask=True)

        x += residue

        ##Feed Forward LAYER

        residue = x

        x = self.layernorm_2(x)

        x = self.linear_1(x)

        x = x * torch.sigmoid(1.702 * x) #quickGELU activation function

        x = self.linear_2(x)

        x += residue

        return x



class CLIP(nn.module):

    def __init__(self):
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.Module([
            CLIPLayer(12, 768) for i in range (12)
        ])

        self.layernorm = nn.LayerNorm(768)


    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        #(batch_size, Sequence_Length) -> (batch_size, Sequence_Length, Embedding_Dimension
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)
        
        # (batch_size, Sequence_Length, Embedding_Dimension) 
        output = self.layernorm(state)

        return output