
import math
import torch
from torch import nn, optim
from torch import Tensor
from typing import Optional, Tuple

# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(query: Tensor,
                                 key: Tensor, 
                                 value: Tensor, 
                                 mask: Optional[Tensor]=None,
                                 dropout_p: Optional[float]=0.0, 
                                 is_causal: Optional[bool]=False,
                                 scale: Optional[float]=None
                                 ) -> Tensor:
    """
    Combine three tensors; query, key, and value; to generate an output tensor of scaled dot product attention.

    Parameters:
    - query (Tensor)              - shape (N x ... x L x E)
    - key (Tensor)                - shape (N x ... x S x E)
    - value (Tensor)              - shape (N x ... x S x Ev)
    - mask (optional Tensor)      - shape (N x ... x L x S)

            mask; shape must be broadcastable to the shape of attention weights.
                            Two types of masks are supported. 
                                1) A boolean mask where a value of True indicates that the element should take part in attention. 
                                2) A float mask of the same type as query, key, value that is added to the attention score.

    - dropout_p (float)           - Dropout probability; if greater than 0.0, dropout is applied

    Returns:
    - Attention output (Tensor)   - shape (N x ... x L x Ev)


    Shape legend:
    - N:    Batch size
    - ...:  Any number of other batch dimensions (optional)
    - S:    Source sequence length
    - L:    Target sequence length
    - E:    Embedding dimension of the query and key
    - Ev:   Embedding dimension of the value
    """
    L, S = query.size(-2), key.size(-2)

    # Calculate scaling factor ahead of time
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale

    # Pre-define attn_bias as zero-weighted tensor
    # this allows it to be included in the attn_weight 
    # calculation regardless of being defined
    attn_bias = torch.zeros(L, S, dtype=query.dtype)

    if is_causal:
        assert mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if mask is not None:
        if mask.dtype == torch.bool:
            attn_bias.masked_fill_(mask.logical_not(), float("-inf"))
        else:
            attn_bias += mask

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value, attn_weight


def expand_mask(mask: torch.Tensor)->torch.Tensor:
    """
    Helper function to support different mask shapes.

    Output shape supports (batch size, number of heads, seq length, seq length)
        If 2D: broadcasted over batch size and number of heads
        If 3D: broadcasted over number of heads
        If 4D: leave as is
    """
    assert mask.ndim >= 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask


class MultiheadAttention(nn.Module):
    """
    Allows the model to jointly attend to information from different representation subspaces.
    Method described in the paper: Attention Is All You Need <https://arxiv.org/abs/1706.03762>
    """
    def __init__(self, 
                 input_dim: int, 
                 embed_dim: int,
                 num_heads: int,
                 dropout_p: Optional[float]=0.0,
                 is_causal: Optional[bool]=False):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_p = dropout_p
        self.is_causal = is_causal

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.query_proj = nn.Linear(input_dim, embed_dim)
        self.key_proj = nn.Linear(input_dim, embed_dim)
        self.value_proj = nn.Linear(input_dim, embed_dim)

        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        """
        Reset the parameters for all tensors defined in def __init__()

            Projection weights are filled with random numbers from the xavier_uniform_ distribution

            Projection biases are filled with floating point zero values

        """
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)

        self.query_proj.bias.data.fill_(0)
        self.key_proj.bias.data.fill_(0)
        self.value_proj.bias.data.fill_(0)
        self.output_proj.bias.data.fill_(0)

    def split_heads(self, 
                    x: Tensor,
                    batch_size: int
                    )->Tensor:
        """
        Split the input tensor into attention heads.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
            Shape: [batch_size, seq_length, embed_dim]
        batch_size : int
            The size of the batch.

        Returns
        -------
        torch.Tensor
            The reshaped input tensor.
            Shape: [batch_size, num_heads, seq_length, head_dim]
        """
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self,
                x: Tensor,
                y: Optional[Tensor] = None,
                attn_mask: Optional[Tensor]=None,
                key_padding_mask: Optional[Tensor]=None,
                return_attention: Optional[bool]=False
                ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass of the multi-head attention module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
            Shape: [batch_size, seq_length, embed_dim]

        y : torch.Tensor, optional
            The optional input tensor for cross-attention.
            Shape: [batch_size, seq_length, embed_dim]

        mask : torch.Tensor, optional
            The mask to be applied to the attention scores.
            Shape: [batch_size, 1, 1, seq_length]

        key_padding_mask : torch.Tensor, optional
            The mask for key padding.
            Shape: [batch_size, seq_length]

        Returns
        -------
        torch.Tensor
            The output tensor of the multi-head attention.
            Shape: [batch size, sequence length, embedding dim]
        """
        batch_size = x.size(0)

        # Combine mask and key_padding_mask if one or both are defined
        # expand_mask ensures that they are all of Shape: [batch_size, 1, 1, seq_length]
        if attn_mask is not None and key_padding_mask is not None:
            mask=expand_mask(mask)
            mask+=expand_mask(key_padding_mask)
        elif attn_mask is not None:
            mask=expand_mask(mask)
        elif key_padding_mask is not None:
            mask=expand_mask(key_padding_mask)
        else:
            mask=None

        # Dot-product of inputs vs query, key and value projections
        q = self.split_heads(self.query_proj(x), batch_size)
        # Cross-attention - Use cross-attention inputs for keys and values
        k = self.split_heads(self.key_proj(y if y is not None else x), batch_size)
        v = self.split_heads(self.value_proj(y if y is not None else x), batch_size)

        # Calculate attention tensor
        attention, attn_weight = scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout_p, is_causal=self.is_causal)

        attention = attention.permute(0, 2, 1, 3).contiguous()
        attention = attention.view(batch_size, -1, self.embed_dim)
        output = self.output_proj(attention)

        if return_attention==True:
            return output, attn_weight
        else:
            return output


class EncoderBlock(nn.Module):
    """
    An encoder block is is made up of a self-attention block and a feedforward network.
    """

    def __init__(self,
                 input_dim: int,
                 num_heads: int,
                 dim_feedforward: int,
                 dropout_p: Optional[float]=0.0):
        """
        Inputs:
            input_dim (int)       - Dimensionality of the input
            num_heads (int)       - Number of heads to use in the attention block
            dim_feedforward (int) - Dimensionality of the hidden layer in the MLP
            dropout_p (float)     - Dropout probability to use in the dropout layers
        """
        super().__init__()
        
        # Attention layer
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads, dropout_p)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout_p),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self,
                x: Tensor,
                key_padding_mask: Optional[Tensor]=None,
                return_attention: Optional[bool]=False
                )->Tensor:

        # Attention part
        x_new = self.norm1(x)
        x_new = self.self_attn(x, key_padding_mask=key_padding_mask, return_attention=return_attention)
        x_new = self.dropout(x_new)
        x = x_new + x

        # MLP part
        x_new = self.norm2(x)
        x_new = self.linear_net(x)
        x_new = self.dropout(x_new)
        x = x_new + x

        return x



class VaswaniDecoderBlock(nn.Module):
    """
    The original decoder block described in Vaswani et al (2017). Consisting of self attention, cross attention, and feed-forward layers.
    """

    def __init__(self,
                 input_dim: int,
                 num_heads: int,
                 dim_feedforward: int,
                 dropout_p: Optional[float]=0.0):
        """
        Inputs:
            input_dim (int)       - Dimensionality of the input
            num_heads (int)       - Number of heads to use in the attention block
            dim_feedforward (int) - Dimensionality of the hidden layer in the MLP
            dropout_p (float)     - Dropout probability to use in the dropout layers
        """
        super().__init__()
        
        # Attention layer
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads, dropout_p, is_causal=True)
        self.cross_attn = MultiheadAttention(input_dim, input_dim, num_heads, dropout_p)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout_p),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.norm3 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self,
                x: Tensor,
                memory: Tensor,
                src_key_padding_mask: Optional[Tensor]=None,
                tgt_key_padding_mask: Optional[Tensor]=None,
                return_attention: Optional[bool]=False
                )->Tensor:
        """
        Forward pass of the Transformer decoder block.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
            Shape: [batch size, sequence length, embedding dim]
        encoder_memory : torch.Tensor
            The memory tensor from the encoder (i.e. output of the encoder).
            Shape: [batch size, sequence length, embedding dim]
        mask : torch.Tensor
            The mask to be applied to the attention scores.
            Shape: [batch size, 1, 1, sequence length]
        src_key_padding_mask : torch.Tensor, optional
            The mask for source key padding.
            Shape: [batch size, sequence length]
        tgt_key_padding_mask : torch.Tensor, optional
            The mask for target key
            Shape: [batch size, sequence length]

        Returns
        -------
        torch.Tensor
            The output tensor of the Transformer encoder block.
            Shape: [batch size, sequence length, embedding dim]
        """

        # Self-attention part
        x_new = self.norm1(x)
        x_new = self.self_attn(x_new, key_padding_mask=tgt_key_padding_mask)
        x_new = self.dropout(x_new)
        x = x_new + x

        # Cross-attention
        x_new = self.norm2(x)
        x_new = self.cross_attn(x_new, memory, key_padding_mask=src_key_padding_mask)
        x_new = self.dropout(x_new)
        x = x_new + x

        # MLP part
        x_new = self.norm3(x)
        x_new = self.linear_net(x_new)
        x_new = self.dropout(x_new)
        x = x_new + x

        return x
    

class GPTDecoderBlock(nn.Module):
    """
    The original decoder block described in Vaswani et al (2017). Consisting of self attention, cross attention, and feed-forward layers.
    """

    def __init__(self,
                 input_dim: int,
                 num_heads: int,
                 dim_feedforward: int,
                 dropout_p: Optional[float]=0.0):
        """
        Inputs:
            input_dim (int)       - Dimensionality of the input
            num_heads (int)       - Number of heads to use in the attention block
            dim_feedforward (int) - Dimensionality of the hidden layer in the MLP
            dropout_p (float)     - Dropout probability to use in the dropout layers
        """
        super().__init__()
        
        # Attention layer
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads, dropout_p, is_causal=True)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout_p),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self,
                x: Tensor,
                memory: Tensor,
                src_key_padding_mask: Optional[Tensor]=None,
                tgt_key_padding_mask: Optional[Tensor]=None,
                return_attention: Optional[bool]=False
                )->Tensor:
        """
        Forward pass of the Transformer decoder block.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
            Shape: [batch size, sequence length, embedding dim]
        encoder_memory : torch.Tensor
            The memory tensor from the encoder (i.e. output of the encoder).
            Shape: [batch size, sequence length, embedding dim]
        mask : torch.Tensor
            The mask to be applied to the attention scores.
            Shape: [batch size, 1, 1, sequence length]
        src_key_padding_mask : torch.Tensor, optional
            The mask for source key padding.
            Shape: [batch size, sequence length]
        tgt_key_padding_mask : torch.Tensor, optional
            The mask for target key
            Shape: [batch size, sequence length]

        Returns
        -------
        torch.Tensor
            The output tensor of the Transformer encoder block.
            Shape: [batch size, sequence length, embedding dim]
        """

        # Self-attention part
        x_new = self.norm1(x)
        x_new = self.self_attn(x_new, key_padding_mask=tgt_key_padding_mask)
        x_new = self.dropout(x_new)
        x = x_new + x

        # Cross-attention
        x_new = self.norm2(x)
        x_new = self.cross_attn(x_new, memory, key_padding_mask=src_key_padding_mask)
        x_new = self.dropout(x_new)
        x = x_new + x

        # MLP part
        x_new = self.norm3(x)
        x_new = self.linear_net(x_new)
        x_new = self.dropout(x_new)
        x = x_new + x

        return x
    

class TransformerBlock(nn.Module):

    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 input_dim: int,
                 num_heads: int,
                 dim_feedforward: int,
                 dropout_p: Optional[float]=0.0
                 ):
        """
        A custom Transformer block, consisting of a stack of encoder and decoder layers.

        Parameters
        ----------
        num_encoder_layers : int
            The number of encoder layers in the Transformer block.
        num_decoder_layers : int
            The number of decoder layers in the Transformer block.
        input_dim : int
            The dimension of the input embeddings.
        num_heads : int
            The number of attention heads.
        dim_feedforward : int
            The inner size of the feed-forward networks in the encoder and decoder layers.
        dropout_p : float, optional, default=0.1
            The dropout rate.

        Attributes
        ----------
        encoder : torch.nn.ModuleList
            A list of TransformerEncoderBlock layers.
        decoder : torch.nn.ModuleList
            A list of TransformerDecoderBlock layers.
        """
        super(TransformerBlock, self).__init__()

        self.encoder = nn.ModuleList([
            EncoderBlock(input_dim=input_dim, nr_attention_heads=num_heads, dim_feedforward=dim_feedforward, dropout_p=dropout_p)
            for i in range(num_encoder_layers)
        ])

        self.decoder = nn.ModuleList([
            DecoderBlock(input_dim=input_dim, nr_attention_heads=num_heads, dim_feedforward=dim_feedforward, dropout_p=dropout_p)
            for i in range(num_decoder_layers)
        ])

    def forward(self, src, tgt, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Forward pass of the custom Transformer block.

        Parameters
        ----------
        src : torch.Tensor
            The input tensor for the encoder.
            Shape: [batch size, sequence length, embedding dim]
        tgt : torch.Tensor
            The input tensor for the decoder.
            Shape: [batch size, sequence length, embedding dim]
        tgt_mask : torch.Tensor, optional
            The mask to be applied to the decoder's attention scores.
            Shape: [batch size, 1, 1, sequence length]
        src_key_latitude, tgt_key_padding_mask : torch.Tensor, optional
            The mask for source/target key padding.
            Shape: [batch size, sequence length]
        memory_key_padding_mask : torch.Tensor, optional
            The mask for memory key padding, currently unused.
            Shape: [batch size, sequence length]

        Returns
        -------
        torch.Tensor
            The output tensor of the custom Transformer block.
            Shape: [batch size, sequence length, embedding dim]
        """
        for idx, enc in enumerate(self.encoder):
            src = enc(src, key_padding_mask=src_key_padding_mask)
        for idx, dec in enumerate(self.decoder):
            tgt = dec(tgt, encoder_memory=src, mask=tgt_mask, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        return tgt

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Generates an upper-triangular matrix of -inf, with zeros on the diagonal.

        Parameters
        ----------
        sz : int
            The size of the square matrix.

        Returns
        -------
        torch.Tensor
            An upper-triangular matrix of -inf, with zeros on the diagonal.
            Shape: [sz, sz]
        """
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)