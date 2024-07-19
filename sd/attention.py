import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        # Projected of the input prior to the multi-head attention
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_heads = d_embed // n_heads

    #Masking is used to prevent the model from looking at the future and instead only past tokens
    def forward(self, x: torch.Tensor, casual_mask=False):
        #x: (Batch_Sizez, Sequence_Length, Embedding_Dimension)

        input_shape = x.shape

        batch_size, sequence_length, d_embed = input_shape

        intermim_shape = (batch_size, sequence_length, self.n_heads, self.d_heads)

        # (Batch_Size, Sequence_Length, Embedding_Dimension) -> (Batch_Size, Sequence_Length, 3 * Embedding_Dimension) -> 3 tensors of shape (Batch_Size, Sequence_Length, Embedding_Dimension)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (Batch_Size, Sequence_Length, Embedding_Dimension) -> (Batch_Size, Sequence_Length, Number of Heads, Embedding Dimension / Head) 
        # -> (Batch_Size, Number of Heads, Sequence_Length, Embedding Dimension / Head)
        q = q.view(intermim_shape).transpose(1,2)
        k = k.view(intermim_shape).transpose(1,2)
        v = v.view(intermim_shape).transpose(1,2)

        #(Batch_Size, Number of Heads, Sequence_Length, Sequence_Length
        weight = q @ k.transpose(-1, -2)

        if casual_mask:
            #Mask where the upper tringal (above the diagonal) is made of ones
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            #Fill it up with minus infinity
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_heads)

        #Softmax applied
        weight = F.softmax(weight, dim=-1)
        #(batch_Size, Number of Heads, Sequence_Length, Sequence_Length) -> (Batch_Size, Number of Heads, Sequence_Length, Embedding Dimension / Head 
        output = weight @ v
        # (Batch_Size, Number of Heads, Sequence_Length, Embedding Dimension / Head  -> (Batch_Size, Sequence_Length, Number of Heads, Embedding Dimension / Head)
        output = output.transpose(1,2)

        output = output.reshape(input_shape)

        output = self.out_proj(output)

        # (Batch_Size, Sequence_Length, Embedding_Dimension) -> (Batch_Size, Sequence_Length, Embedding_Dimension)
        return output


