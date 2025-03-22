import io
import math
import torch
import torch.nn as nn
import numpy as np
import chess.pgn
from chess import Board
import torch.nn.functional as F

PIECE_CHARS = "♔♕♖♗♘♙⭘♟♞♝♜♛♚"

def encode_board(board: Board) -> np.array:
    # String-encode the board.
    # If board.turn = 1 then it is now white's turn which means this is a potential move
    # being contemplated by black, and therefore we reverse the char order to rotate the board
    # for black's perspective
    # If board.turn = 0 then it is now black's turn which means this is a potential move
    # being contemplated by white, and therefore we leave the char order as white's perspective.
    # Also reverse PIECE_CHARS indexing order if black's turn to reflect "my" and "opponent" pieces.
    step = 1 - 2 * board.turn
    unicode = board.unicode().replace(' ','').replace('\n','')[::step]
    return np.array([PIECE_CHARS[::step].index(c) for c in unicode], dtype=int).reshape(8,8)

class Attention(nn.Module):
    def __init__(self, input_dims, attention_dims, n_heads=2, use_flash_attn=True, dropout=0.1):
        super().__init__()
        assert attention_dims % n_heads == 0, "attention_dims must be divisible by n_heads"
        self.attention_dims = attention_dims
        self.n_heads = n_heads
        self.head_dim = attention_dims // n_heads
        # Check if Flash Attention is available
        self.use_flash_attn = use_flash_attn and hasattr(F, 'scaled_dot_product_attention')
        # Combined QKV projection for efficiency (reduces parameter count and computation)
        self.qkv_proj = nn.Linear(input_dims, 3 * attention_dims)
        # Combined QKV projection for efficiency (reduces parameter count and computation)
        self.qkv_proj = nn.Linear(input_dims, 3 * attention_dims)
        # Output projection
        self.o_proj = nn.Linear(attention_dims, input_dims)
        # Normalization and regularization
        self.norm = nn.LayerNorm(input_dims)
        self.dropout = nn.Dropout(dropout)
        # Initialize parameters properly for stable training
        self._reset_parameters()

    def _reset_parameters(self):
        # Xavier uniform initialization for stable gradient flow
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
 
    def forward(self,x):
        '''
        x: shape (B,D,k1,k2) where B is the Batch size, D is number of filters, and k1, k2 are the kernel sizes
        '''
        # Save original dimensions for reshaping later
        oB, oD, oW, oH = x.shape
        seq_len = oW * oH
        # Reshape to sequence form: (B, D, k1, k2) -> (B, k1*k2, D)
        x_reshaped = x.permute(0, 2, 3, 1).reshape(oB, seq_len, oD)
        # Store residual for skip connection
        residual = x_reshaped
        # Apply layer normalization (pre-norm architecture)
        x_norm = self.norm(x_reshaped)
        # Project to queries, keys, values all at once (more efficient)
        qkv = self.qkv_proj(x_norm)
        qkv = qkv.reshape(oB, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Separate Q, K, V
        # Apply attention mechanism
        if self.use_flash_attn:
            # Use Flash Attention via PyTorch's optimized implementation
            attn_output = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout.p if self.training else 0.0
            )
        else:
            # Calculate attention scores
            scale = 1.0 / math.sqrt(self.head_dim)
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn) if self.training else attn
            attn_output = torch.matmul(attn, v)
        # Reshape attention output back to sequence format
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(oB, seq_len, self.attention_dims)
        # Project back to input dimension
        output = self.o_proj(attn_output)
        output = self.dropout(output)
        # Add residual connection
        output = output + residual
        # Reshape back to original spatial dimensions: (B, seq_len, D) -> (B, D, k1, k2)
        output = output.reshape(oB, oW, oH, oD).permute(0, 3, 1, 2)
        return output

class Residual(nn.Module):
    """
    The Residual block of ResNet models.
    """
    def __init__(self, outer_channels, inner_channels, use_1x1conv, dropout, dilation = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(outer_channels, inner_channels, kernel_size=3, padding='same', stride=1, dilation = dilation)
        self.conv2 = nn.Conv2d(inner_channels, outer_channels, kernel_size=3, padding='same', stride=1, dilation = dilation)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(outer_channels, outer_channels, kernel_size=1, stride=1)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(inner_channels)
        self.bn2 = nn.BatchNorm2d(outer_channels)
        self.dropout = nn.Dropout(p=dropout)
        # GELU activation
        self.gelu = nn.GELU()
        # Initialize parameters using Kaiming normal
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Use 'relu' for initialization, then scale for GELU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                m.weight.data *= 1.0 / 0.7978845608028654  # Scaling factor for GELU
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, X):
        Y = self.gelu(self.bn1(self.conv1(X)))
        Y = self.dropout(self.bn2(self.conv2(Y)))
        # 1x1 convolution if needed (using memory-efficient implementation)
        if self.conv3 is not None:
            X = self.conv3(X)
        Y += X
        return F.gelu(Y)

class Model(nn.Module):
    """
    Convolutional Model
    Note: the 'device' argument is not used, only included to simplify the repo overall.
    """
    def __init__(self, nlayers, embed_dim, inner_dim, attention_dim, use_1x1conv, dropout, device='cpu'):
        super().__init__()
        self.vocab = PIECE_CHARS
        self.embed_dim = embed_dim
        self.inner_dim = inner_dim
        self.use_1x1conv = use_1x1conv
        self.dropout = dropout

        self.embedder = nn.Embedding(len(self.vocab), self.embed_dim)
        self.convLayers = nn.ModuleList()
        for i in range(nlayers): 
            self.convLayers.append(Residual(self.embed_dim, self.inner_dim, self.use_1x1conv, self.dropout, 2**i))
            self.convLayers.append(Attention(self.embed_dim, attention_dim))

        self.convnet = nn.Sequential(*self.convLayers)
        self.accumulator = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=8, padding=0, stride=1)
        self.decoder = nn.Linear(self.embed_dim, 1)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.embedder.weight, -initrange, initrange)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, inputs):
        inputs = self.embedder(inputs)
        inputs = torch.permute(inputs, (0, 3, 1, 2)).contiguous() 
        inputs = self.convnet(inputs)
        inputs = F.relu(self.accumulator(inputs).squeeze())
        scores = self.decoder(inputs).flatten()
        return scores

    def score(self, pgn, move):
        '''
        pgn: string e.g. "1.e4 a6 2.Bc4 "
        move: string e.g. "a5 "
        '''
        # init a game and board
        game = chess.pgn.read_game(io.StringIO(pgn))
        board = Board()
        # catch board up on game to present
        for past_move in list(game.mainline_moves()):
            board.push(past_move)
        # push the move to score
        board.push_san(move)
        # convert to tensor, unsqueezing a dummy batch dimension
        board_tensor = torch.tensor(encode_board(board)).unsqueeze(0)
        return self.forward(board_tensor).item()