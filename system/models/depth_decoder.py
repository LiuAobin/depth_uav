import torch
from torch import nn


class FullQueryLayer(nn.Module):
    def __init__(self) -> None:
        super(FullQueryLayer, self).__init__()
    def forward(self, x, K):
        """
        given feature map of size [bs, E, H, W], and queries of size [bs, Q, E]
        return Q energy maps corresponding to Q queries of shape [bs, Q, H, W]
        and add feature noise to x of the same shape as input [bs, E, H, W]
        and summary_embedding of shape [bs, Q, E]
        计算self-cost volume V
        Args:
            x: feature map S of shape (B, E, H, W)
            K: queries Q of shape (B, Q, E) Q is query_num的数量
        Returns:
            energy map: Q 个查询对应的能量图 of shape (B, Q, H,W)
            summary_embedding: 总结后的嵌入表示 of shape (B, Q, E)
        """
        n, c, h, w = x.size() # bs, E, H, W
        _, cout, ck = K.size() # bs, Q, E
        assert c == ck, "Number of channels in x and Embedding dimension (at dim 2) of K matrix must match"
        # 计算查询对特征图的相似度
        # 先将x变换为形状(B,H*W,E)(展平空间维度)
        # 进行矩阵乘法，得到[B,H*W,Q]的能量分布
        y = torch.matmul(x.view(n, c, h * w).permute(0, 2, 1),
                         K.permute(0, 2, 1))
        y_norm = torch.softmax(y, dim=1)
        # 计算总结嵌入 summary_embedding
        # 1. 交换 y_norm 的维度，形状变为 [B, Q, H*W]
        # 2. 计算权重和输入特征的加权和，得到形状 [B, Q, E] 的总结嵌入
        summary_embedding = torch.matmul(y_norm.permute(0, 2, 1),
                                         x.view(n, c, h*w).permute(0, 2, 1))
        y = y.permute(0, 2, 1).view(n, cout, h, w)
        return y, summary_embedding


class DepthDecoderQueryTr(nn.Module):
    def __init__(self, in_channels,# C
                 embedding_dim=128, patch_size=16, # E _
                 num_heads=4, query_nums=100,# _ Q
                 dim_out=256, norm='linear',# _ _
                 min_val=0.001, max_val=10) -> None:
        super(DepthDecoderQueryTr, self).__init__()
        self.norm = norm

        # Corase-grained queries Q
        # get a feature map F of shape C x h/p x h/p
        self.embedding_convPxP = nn.Conv2d(in_channels, embedding_dim,
                                           kernel_size=patch_size, stride=patch_size, padding=0)
        self.positional_encodings = nn.Parameter(torch.rand(500, embedding_dim),
                                                 requires_grad=True)

        # mini-transformer of 4 layers to generate a set coarse-grained queries Q of shape R^{CxQ}
        encoder_layers = nn.modules.transformer.TransformerEncoderLayer(
            embedding_dim, num_heads, dim_feedforward=1024)
        # encoder_layers = nn.modules.transformer.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward=512)
        self.transformer_encoder = nn.modules.transformer.TransformerEncoder(
            encoder_layers, num_layers=4)

        # get self-cost volume V
        self.conv3x3 = nn.Conv2d(in_channels, embedding_dim,
                                 kernel_size=3, stride=1, padding=1)

        # self.summary_layer = PixelWiseDotProduct_for_summary()
        # self.dense_layer = PixelWiseDotProduct_for_dense()
        self.full_query_layer = FullQueryLayer()

        # a MLP to regress the depth bins b
        self.bins_regressor = nn.Sequential(nn.Linear(embedding_dim*query_nums,
                                                      16*query_nums),
                                       nn.LeakyReLU(),
                                       nn.Linear(16*query_nums, 16*16),
                                       nn.LeakyReLU(),
                                       nn.Linear(16*16, dim_out))
        # probabilistic combination
        # a 1x1 convolution to the self volume V to obtain a D-planes volumes
        # a plane-wise softmax operation to convert the volume into plane-wise probabilistic map
        self.convert_to_prob = nn.Sequential(nn.Conv2d(query_nums, dim_out,
                                                       kernel_size=1, stride=1, padding=0),
                                      nn.Softmax(dim=1))
        self.query_nums = query_nums

        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x0):
        """
       Args:
           x0: high resolution immediate features S of shape [B, C, H, W]
       Returns:
       """
        # 1. 计算粗粒度查询Q
        # apply a convolution of kernel size pxp and stride=p to S get a feature map F of shape (B,E,h/p,w/p)
        embeddings_0 = self.embedding_convPxP(x0.clone())  # [B,E,H/p W/p]
        # reshape F to (B,E,N)
        embeddings_0 = embeddings_0.flatten(2) # [B,E,N],其中N=(H/p W/p)
        # add positional embeddings to F
        embeddings_0 = embeddings_0 + self.positional_encodings[:embeddings_0.shape[2], :].T.unsqueeze(0) #  [B,E,N]
        embeddings_0 = embeddings_0.permute(2, 0, 1)  # reshape F to (N,B,E)方便后边计算
        # feed these patch embeddings into a mini-transformer of 4 layers to generate as set of coarse-grained queries Q of shape R^{C,Q}
        total_queries = self.transformer_encoder(embeddings_0)  # [N,B,E]

        x0 = self.conv3x3(x0) # [B,E,H,W]
        # get self-cost volume V = total_queries^T \cdot S
        queries = total_queries[:self.query_nums, ...]  # [Q, B, E],其中Q=query_nums
        queries = queries.permute(1, 0, 2) # [B,Q,E]

        # summarys用于计算箱子宽度，energy_maps用于计算像素落在该箱子的概率
        # energy_maps=[B,Q,H,W]  summarys=[B,Q,E]
        energy_maps, summarys = self.full_query_layer(x0, queries)
        bs, Q, E = summarys.shape
        # 2. 预测深度bins
        y = self.bins_regressor(summarys.view(bs, Q * E))

        # 3. 归一化深度分箱
        if self.norm == 'linear':
            y = torch.relu(y)
            eps = 0.1
            y = y + eps
        elif self.norm == 'softmax':
            return torch.softmax(y, dim=1), energy_maps
        else:
            y = torch.sigmoid(y)
        y = y / y.sum(dim=1, keepdim=True)

        # 4. 计算最终的深度估计
        # 转换自成本体积为概率
        out = self.convert_to_prob(energy_maps)  # [B,dim_out,H,W]
        bin_widths = (self.max_val - self.min_val) * y # [B,dim_out]
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val) # [B, dim_out+1]
        bin_edges = torch.cumsum(bin_widths, dim=1) # [B, dim_out+1]

        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])   # [B, dim_out]
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1)  # [B, dim_out, 1, 1]

        pred = torch.sum(out * centers, dim=1, keepdim=True)  # [B, 1, H, W]
        # outputs = {"depth": pred, "bins": bin_edges}
        # return outputs
        return pred