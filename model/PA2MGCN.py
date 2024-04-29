from layers.PA2GCNrelated import *
from layers.ST_Nom_layer import *
from torch_utils.graph_process import *
import torch
import torch.nn as nn
import torch.nn.functional as F


device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')
class PA2MGCN(nn.Module):
    def __init__(self, num_nodes, seq_len=12,num_features=3,pred_len=12,supports=None,dropout=0.3,residual_channels=32, dilation_channels=32,
                 skip_channels=256, end_channels=512, kernel_size=2, K=3, Kt=3,**kwargs):
        super(PA2MGCN, self).__init__()
        # Changing variable names
        length=seq_len
        in_dim=num_features
        out_dim=pred_len
        args=kwargs.get('args')
        self.args=args
        self.dropout = dropout
        self.num_nodes = num_nodes
        self.in_channels = num_features

        self.bn = nn.BatchNorm2d(in_dim, affine=False)
        self.time_of_day_size = args.time_of_day_size
        self.day_of_week_size = args.day_of_week_size
        self.if_time_in_day = args.if_T_i_D
        self.if_day_in_week = args.if_D_i_W
        self.if_spatial = args.if_node
        self.embed_dim = args.d_model
        self.node_dim = self.embed_dim
        self.temp_dim_tid = self.embed_dim
        self.temp_dim_diw = self.embed_dim

        '''Embedding Layer'''
        # spatial embeddings
        if self.if_spatial:
            self.node_emb = nn.Parameter(
                torch.empty(args.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)
        # temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=args.num_features * args.seq_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)
        # encoding
        self.hidden_dim = self.embed_dim + self.node_dim * \
                          int(self.if_spatial) + self.temp_dim_tid * int(self.if_time_in_day) + \
                          self.temp_dim_diw * int(self.if_day_in_week)
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(args.num_layer)])

        self.emb_conv = nn.Conv2d(self.in_channels, self.hidden_dim, kernel_size=(1, 1))
        self.in_conv = nn.Conv2d(self.hidden_dim, residual_channels, kernel_size=(1, 1))

        self.start_conv = nn.Conv2d(in_channels=self.hidden_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        # Handling the adjacency matrix
        supports = calculate_laplacian_with_self_loop(supports)
        self.supports = [supports]

        self.supports_len = 0

        if supports is not None:
            self.supports_len += len(self.supports)

        if supports is None:
            self.supports = []
        # learnable adj
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
        self.h = nn.Parameter(torch.zeros(num_nodes, num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)
        self.supports_len += 1

        # Norm Layer1
        t_norm1=TNorm(num_nodes, dilation_channels)
        s_norm1=SNorm(dilation_channels)

        # ST-Blcok1
        self.block1 = ST_Blcok(dilation_channels, dilation_channels, num_nodes=num_nodes,tem_size= length - 6,Kt=3, dropout=dropout,
                              support_len=self.supports_len,args=kwargs.get('args'),t_norm=t_norm1,s_norm=s_norm1)

        # Norm Layer2
        t_norm2=TNorm(num_nodes, dilation_channels)
        s_norm2=SNorm(dilation_channels)

        # ST-Blcok2
        self.block2 = ST_Blcok(dilation_channels, dilation_channels, num_nodes=num_nodes,tem_size=length - 9, Kt=2,dropout=dropout,
                              support_len=self.supports_len,args=kwargs.get('args'),t_norm=t_norm2,s_norm=s_norm2)

        # residual connect
        self.skip_conv1 = nn.Conv2d(dilation_channels, skip_channels, kernel_size=(1, 1),
                                 stride=(1, 1), bias=True)
        self.skip_conv2 = nn.Conv2d(dilation_channels, skip_channels, kernel_size=(1, 1),
                                 stride=(1, 1), bias=True)

        # Output Layer
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 3),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)



    def forward(self, input,adj,**kwargs):
        # input(B,C,N,L),return:(B,C,N,L)
        input = self.bn(input)
        x=input.clone()
        input_data = input.permute(0, 3, 2, 1)  # [B, L, N, C]
        input_data = input_data[..., range(self.in_channels)]
        seq_time = kwargs.get('seqs_time')

        '''Embedding Layer'''
        # time(dayofyear, dayofmonth, dayofweek, hourofday, minofhour)
        if self.if_time_in_day:
            hour = (seq_time[:, -2:-1, ...] + 0.5) * 23 #To get the hour
            min = (seq_time[:, -1:, ...] + 0.5) * 59  # To Get the minute.
            hour_index = (hour * 60 + min) / (60 / self.args.points_per_hour)
            time_in_day_emb = self.time_in_day_emb[
                hour_index[..., -1].squeeze(1).repeat(1, self.args.num_nodes).type(torch.LongTensor)]  # (B,N,D)
        else:
            time_in_day_emb = None
        if self.if_day_in_week:
            day = (seq_time[:, 2:3, ...] + 0.5) * (6 - 0)
            day_in_week_emb = self.day_in_week_emb[
                day[..., -1].squeeze(1).repeat(1, self.args.num_nodes).type(torch.LongTensor)]
        else:
            day_in_week_emb = None
        # time series embedding
        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(
            batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(input_data)
        node_emb = []
        if self.if_spatial:
            # expand node embeddings
            node_emb.append(self.node_emb.unsqueeze(0).expand(
                batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))

        # temporal embeddings
        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))

        # concate all embeddings
        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)
        hidden = self.encoder(hidden)
        x = self.emb_conv(x) + hidden

        if self.supports is not None:
            A = F.relu(torch.mm(self.nodevec1, self.nodevec2)) # trainable adjacency matrix
            d = 1 / (torch.sum(A, -1)) # Normalize
            D = torch.diag_embed(d)
            A = torch.matmul(D, A)
            if not isinstance(self.supports,list):
                self.supports=[self.supports]
            new_supports = self.supports + [A.to(device)]

        '''Stacked Spatial-Temporal Blocks'''
        skip = 0
        x = self.start_conv(x)

        # S-T block1
        x = self.block1(x, new_supports)

        s1 = self.skip_conv1(x)
        skip = s1 + skip

        # S-T block2
        x = self.block2(x, new_supports)

        s2 = self.skip_conv2(x)
        skip = skip[:, :, :, -s2.size(3):]
        skip = s2 + skip

        '''Output Layer'''
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        x=x.transpose(1,3)
        return x  # output = [batch_size,1=dim,num_nodes,12=pred_len]
