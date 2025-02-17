import torch
import torch.nn as nn
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.module import Module
from torch.autograd import Variable
import math
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, L, N):

        super(Encoder, self).__init__()

        self.L = L

        self.N = N

        self.Conv1d = nn.Conv1d(in_channels=1,
                                out_channels=N,
                                kernel_size=L,
                                stride=L//2,
                                padding=0,
                                bias=False)

        self.ReLU = nn.ReLU()

    def forward(self, x):

        x = self.Conv1d(x)

        x = self.ReLU(x)

        return x


class Decoder(nn.Module):

    def __init__(self, L, N):

        super(Decoder, self).__init__()

        self.L = L

        self.N = N

        self.ConvTranspose1d = nn.ConvTranspose1d(in_channels=N,
                                                  out_channels=1,
                                                  kernel_size=L,
                                                  stride=L//2,
                                                  padding=0,
                                                  bias=False)

    def forward(self, x):

        x = self.ConvTranspose1d(x)

        return x


class TransformerEncoderLayer(Module):
    """
        TransformerEncoderLayer is made up of self-attn and feedforward network.
        This standard encoder layer is based on the paper "Attention Is All You Need".
        Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
        Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
        Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
        in a different way during application.

        Args:
            d_model: the number of expected features in the input (required).
            nhead: the number of heads in the multiheadattention models (required).
            dim_feedforward: the dimension of the feedforward network model (default=2048).
            dropout: the dropout value (default=0.1).
            activation: the activation function of intermediate layer, relu or gelu (default=relu).

        Examples:
            >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
            >>> src = torch.rand(10, 32, 512)
            >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dropout=0):

        super(TransformerEncoderLayer, self).__init__()

        self.LayerNorm1 = nn.LayerNorm(normalized_shape=d_model)

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        self.Dropout1 = nn.Dropout(p=dropout)

        self.LayerNorm2 = nn.LayerNorm(normalized_shape=d_model)

        self.FeedForward = nn.Sequential(nn.Linear(d_model, d_model*2*2),
                                         nn.ReLU(),
                                         nn.Dropout(p=dropout),
                                         nn.Linear(d_model*2*2, d_model))

        self.Dropout2 = nn.Dropout(p=dropout)

    def forward(self, z):

        z1 = self.LayerNorm1(z)

        z2 = self.self_attn(z1, z1, z1, attn_mask=None, key_padding_mask=None)[0]

        z3 = self.Dropout1(z2) + z

        z4 = self.LayerNorm2(z3)

        z5 = self.Dropout2(self.FeedForward(z4)) + z3

        return z5


class Positional_Encoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):

        super(Positional_Encoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)  # seq_len, batch, channels
        pe = pe.transpose(0, 1).unsqueeze(0)  # batch, channels, seq_len

        self.register_buffer('pe', pe)

    def forward(self, x):

        x = x.permute(0, 2, 1).contiguous()

        # x is seq_len, batch, channels
        # x = x + self.pe[:x.size(0), :]

        # x is batch, channels, seq_len
        x = x + self.pe[:, :, :x.size(2)]

        x = self.dropout(x)

        x = x.permute(0, 2, 1).contiguous()

        return x


class DPTBlock(nn.Module):

    def __init__(self, input_size, nHead, Local_B):

        super(DPTBlock, self).__init__()

        self.Local_B = Local_B

        self.intra_PositionalEncoding = Positional_Encoding(d_model=input_size, max_len=32000)
        self.intra_transformer = nn.ModuleList([])
        for i in range(self.Local_B):
            self.intra_transformer.append(TransformerEncoderLayer(d_model=input_size,
                                                                  nhead=nHead,
                                                                  dropout=0.1))

        self.inter_PositionalEncoding = Positional_Encoding(d_model=input_size, max_len=32000)
        self.inter_transformer = nn.ModuleList([])
        for i in range(self.Local_B):
            self.inter_transformer.append(TransformerEncoderLayer(d_model=input_size,
                                                                  nhead=nHead,
                                                                  dropout=0.1))

    def forward(self, z):

        B, N, K, P = z.shape

        # intra DPT
        row_z = z.permute(0, 3, 2, 1).contiguous().view(B*P, K, N)
        row_z1 = self.intra_PositionalEncoding(row_z)

        for i in range(self.Local_B):
            row_z1 = self.intra_transformer[i](row_z1.permute(1, 0, 2).contiguous()).permute(1, 0, 2).contiguous()

        row_f = row_z1 + row_z
        row_output = row_f.view(B, P, K, N).permute(0, 3, 2, 1).contiguous()

        # inter DPT
        col_z = row_output.permute(0, 2, 3, 1).contiguous().view(B*K, P, N)
        col_z1 = self.inter_PositionalEncoding(col_z)

        for i in range(self.Local_B):
            col_z1 = self.inter_transformer[i](col_z1.permute(1, 0, 2).contiguous()).permute(1, 0, 2).contiguous()

        col_f = col_z1 + col_z
        col_output = col_f.view(B, K, P, N).permute(0, 3, 1, 2).contiguous()

        return col_output


class Separator(nn.Module):

    def __init__(self, N, H, K, Global_B, Local_B):

        super(Separator, self).__init__()

        self.N = N
        self.K = K
        self.Global_B = Global_B
        self.Local_B = Local_B

        self.LayerNorm = nn.LayerNorm(self.N)
        self.Linear1 = nn.Linear(in_features=self.N, out_features=self.N, bias=None)

        self.SepFormer = nn.ModuleList([])
        for i in range(self.Global_B):
            self.SepFormer.append(DPTBlock(N, H, self.Local_B))

        self.PReLU = nn.PReLU()
        self.Conv2d = nn.Conv2d(N, N, kernel_size=1)

        self.output = nn.Sequential(nn.Conv1d(N, N, 1), nn.Tanh())
        self.output_gate = nn.Sequential(nn.Conv1d(N, N, 1), nn.Sigmoid())

    def forward(self, x):

        # Norm + Linear
        x = self.LayerNorm(x.permute(0, 2, 1).contiguous())
        x = self.Linear1(x).permute(0, 2, 1).contiguous()

        # Chunking
        out, gap = self.split_feature(x, self.K)

        # SepFormer
        for i in range(self.Global_B):
            out = self.SepFormer[i](out)

        out = self.Conv2d(self.PReLU(out))

        out = self.merge_feature(out, gap)

        out = F.relu(self.output(out)*self.output_gate(out))
        out = F.relu(out)

        return out

    def pad_segment(self, input, segment_size):


        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2

        rest = segment_size - (segment_stride + seq_len % segment_size) % segment_size

        if rest > 0:
            pad = Variable(torch.zeros(batch_size, dim, rest)).type(input.type())
            input = torch.cat([input, pad], 2)

        pad_aux = Variable(torch.zeros(batch_size, dim, segment_stride)).type(input.type())

        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest

    def split_feature(self, input, segment_size):

        input, rest = self.pad_segment(input, segment_size)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2

        segments1 = input[:, :, :-segment_stride].contiguous().view(batch_size, dim, -1, segment_size)
        segments2 = input[:, :, segment_stride:].contiguous().view(batch_size, dim, -1, segment_size)
        segments = torch.cat([segments1, segments2], 3).view(batch_size, dim, -1, segment_size).transpose(2, 3).contiguous()

        return segments, rest

    def merge_feature(self, input, rest):

        batch_size, dim, segment_size, _ = input.shape
        segment_stride = segment_size // 2
        input = input.transpose(2, 3).contiguous().view(batch_size, dim, -1, segment_size * 2)  # B, N, K, L

        input1 = input[:, :, :, :segment_size].contiguous().view(batch_size, dim, -1)[:, :, segment_stride:]
        input2 = input[:, :, :, segment_size:].contiguous().view(batch_size, dim, -1)[:, :, :-segment_stride]

        output = input1 + input2

        if rest > 0:
            output = output[:, :, :-rest]

        return output.contiguous()


class Sepformer(nn.Module):
    """
        Args:
            N: Number of filters in autoencoder
            L: Length of the filters in autoencoder
            H: Multi-head
            K: segment size
            R: Number of repeats
    """

    def __init__(self, N=64, L=4, H=4, K=250, Global_B=2, Local_B=4):

        super(Sepformer, self).__init__()

        self.N = N
        self.L = L 
        self.H = H 
        self.K = K
        self.Global_B = Global_B
        self.Local_B = Local_B

        self.encoder = Encoder(self.L, self.N)

        self.separator = Separator(self.N, self.H, self.K, self.Global_B, self.Local_B)

        self.decoder = Decoder(self.L, self.N)

    def forward(self, x):

        # Encoding
        x, rest = self.pad_signal(x)

        enc_out = self.encoder(x)

        # Mask estimation
        masks = self.separator(enc_out)

        # Masking
        out = masks * enc_out

        # Decoding
        audio = self.decoder(out)

        audio = audio[:, :, self.L // 2:-(rest + self.L // 2)].contiguous()  # B, 1, T
        audio = audio.squeeze(1)  # B, T

        return audio

    def pad_signal(self, input):

        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")

        if input.dim() == 2:
            input = input.unsqueeze(1)

        batch_size = input.size(0)
        nsample = input.size(2)
        rest = self.L - (self.L // 2 + nsample % self.L) % self.L

        if rest > 0:
            pad = Variable(torch.zeros(batch_size, 1, rest)).type(input.type())
            input = torch.cat([input, pad], dim=2)

        pad_aux = Variable(torch.zeros(batch_size, 1, self.L//2)).type(input.type())

        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest


if __name__ == "__main__":

    x = torch.rand(1, 16000)

    model = Sepformer(N=128,
                      L=2,
                      H=8,
                      K=250,
                      Global_B=1,
                      Local_B=1)

    print("{:.3f} million".format(sum([param.nelement() for param in model.parameters()]) / 1e6))

    y = model(x)

    print(y.shape)