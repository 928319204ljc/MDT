import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class MAAF(nn.Module):
    """
    Multi-Anchor Adaptive Fusion module for capturing temporal dynamics across multiple time scales.
    """
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 15]):
        super(MAAF, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=(k, 1), padding=(k // 2, 0))
            for k in kernel_sizes
        ])
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Apply multiple convolutions with different kernel sizes
        outputs = [conv(x) for conv in self.convs]
        # Average fusion of different scales
        out = sum(outputs) / len(outputs)
        return self.relu(self.bn(out))



class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A.cuda(x.get_device())
        A = A + self.PA

        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1 + A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x_q, x_k, x_v):
        # 调整输入形状，使其与多头注意力模块兼容
        N, C, T, V = x_q.size()
        x_q = x_q.permute(0, 3, 2, 1).contiguous().view(N * V, T, C)  # [N * V, T, C]
        x_k = x_k.permute(0, 3, 2, 1).contiguous().view(N * V, T, C)
        x_v = x_v.permute(0, 3, 2, 1).contiguous().view(N * V, T, C)

        attn_output, _ = self.attn(x_q, x_k, x_v)  # 输出形状 [N * V, T, C]
        attn_output = attn_output.view(N, V, T, C).permute(0, 3, 2, 1).contiguous()  # 转换回 [N, C, T, V]
        return attn_output


class BGA(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_heads):
        super(BGA, self).__init__()
        self.intra_attention = MultiHeadAttention(in_channels, num_heads)
        self.inter_attention = MultiHeadAttention(in_channels, num_heads)
        self.ffn_intra = nn.Linear(in_channels, hidden_channels)
        self.ffn_inter = nn.Linear(in_channels, hidden_channels)
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        self.relu = nn.ReLU()

    def forward(self, x):
        # Intra-cluster Attention
        x_intra = self.intra_attention(x, x, x)
        N, C, T, V = x_intra.size()  # 获取当前张量的大小

        # 调整形状以匹配线性层的输入要求
        x_intra = x_intra.view(N * V, T, C)  # [N*V, T, C]
        x_intra = self.ffn_intra(x_intra)  # 线性层的输入形状: [N*V, T, in_channels]，输出形状: [N*V, T, hidden_channels]
        x_intra = x_intra.view(N, V, T, -1).permute(0, 3, 2, 1).contiguous()  # 转换回 [N, hidden_channels, T, V]

        # Inter-cluster Attention
        x_inter = self.inter_attention(x, x, x)
        x_inter = x_inter.view(N * V, T, C)
        x_inter = self.ffn_inter(x_inter)
        x_inter = x_inter.view(N, V, T, -1).permute(0, 3, 2, 1).contiguous()

        # Pooling and concatenation
        x_combined = self.pool(x_inter + x_intra)
        return self.relu(x_combined)



class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)
# class fusion(nn.Module):
#     def __init__(self,in_channels,out_channel):
#         super(fusion,self).__init__()
#         self.conv_out=nn.Conv2d(in_channels,out_channel,1, bias=False)
#     def forward(self,x_p,x_m):
#         x=torch.cat((x_p,x_m),1)
#
#         x=self.conv_out(x)
#         return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # 双路径池化：最大池化和平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 计算中间层通道数，至少为1
        hidden_planes = max(1, in_planes // ratio)

        # 共享的 MLP 层
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(hidden_planes, in_planes, 1, bias=False)

        # 使用 Sigmoid 输出注意力权重
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 最大池化路径
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # 平均池化路径
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # 结合两条路径的输出并应用 Sigmoid
        out = self.sigmoid(max_out + avg_out)
        return out


    

class fusion(nn.Module):
    def __init__(self, T_in_channels):
        super(fusion, self).__init__()
        self.att_T_p = ChannelAttention(T_in_channels)
        self.att_N_p = ChannelAttention(16)
        self.att_T_m = ChannelAttention(T_in_channels)
        self.att_N_m = ChannelAttention(16)

    def forward(self, x_p, x_m):
        B, C, T, N = x_p.size()
        x_p_T = x_p.permute(0, 2, 1, 3)
        x_p_N = x_p.permute(0, 3, 2, 1)
        x_m_T = x_m.permute(0, 2, 1, 3)
        x_m_N = x_m.permute(0, 3, 2, 1)

        att_N_p_map = (self.att_N_p(x_p_N)).permute(0, 3, 2, 1)
        x_p_mid = (x_p * att_N_p_map).permute(0, 2, 1, 3)
        att_T_p_map = (self.att_T_p(x_p_mid)).permute(0, 2, 1, 3)

        att_N_m_map = (self.att_N_m(x_m_N)).permute(0, 3, 2, 1)
        x_m_mid = (x_m * att_N_m_map).permute(0, 2, 1, 3)
        att_T_m_map = (self.att_T_m(x_m_mid)).permute(0, 2, 1, 3)

        x_p = x_p + x_m * att_T_m_map
        x_m = x_m + x_p * att_T_p_map

        return x_p, x_m



class Model(nn.Module):
    def __init__(self, num_class=4, num_point=16, num_constraints=31, graph=None, graph_args=dict(), in_channels_p=3, in_channels_m=8):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.data_bn_p = nn.BatchNorm1d(in_channels_p * num_point)
        self.data_bn_m = nn.BatchNorm1d(in_channels_m * num_point)

        # Adding MAAF before TCN-GCN units
        self.maaf_p = MAAF(in_channels_p, 64)  # For input x_p
        self.maaf_m = MAAF(in_channels_m, 64)  # For input x_m

        self.l1_p = TCN_GCN_unit(64, 64, A, residual=False)
        self.l1_m = TCN_GCN_unit(64, 64, A, residual=False)

        self.l2_p = TCN_GCN_unit(64, 64, A)
        self.l3_p = TCN_GCN_unit(64, 64, A)
        self.l4_p = TCN_GCN_unit(64, 64, A)
        self.l5_p = TCN_GCN_unit(64, 128, A, stride=2)
        self.l6_p = TCN_GCN_unit(128, 128, A)
        self.l7_p = TCN_GCN_unit(128, 128, A)
        self.l8_p = TCN_GCN_unit(128, 256, A, stride=2)
        self.l9_p = TCN_GCN_unit(256, 256, A)
        self.l10_p = TCN_GCN_unit(256, 256, A)

        self.l2_m = TCN_GCN_unit(64, 64, A)
        self.l3_m = TCN_GCN_unit(64, 64, A)
        self.l4_m = TCN_GCN_unit(64, 64, A)
        self.l5_m = TCN_GCN_unit(64, 128, A, stride=2)
        self.l6_m = TCN_GCN_unit(128, 128, A)
        self.l7_m = TCN_GCN_unit(128, 128, A)
        self.l8_m = TCN_GCN_unit(128, 256, A, stride=2)
        self.l9_m = TCN_GCN_unit(256, 256, A)
        self.l10_m = TCN_GCN_unit(256, 256, A)

        # BGA Module
        self.bga = BGA(in_channels=256, hidden_channels=256, num_heads=4)

        self.fusion1 = fusion(48)
        self.fusion2 = fusion(24)
        self.fusion3 = fusion(12)

        self.dropout = nn.Dropout(p=0.5)  # Dropout layer with 50% probability

        self.fc1_classifier_p = nn.Linear(256, num_class)
        self.fc1_classifier_m = nn.Linear(256, num_class)
        self.fc2_aff = nn.Linear(256, num_constraints*48)

        nn.init.normal_(self.fc1_classifier_m.weight, 0, math.sqrt(2. / num_class))
        nn.init.normal_(self.fc1_classifier_p.weight, 0, math.sqrt(2. / num_class))
        nn.init.normal_(self.fc2_aff.weight, 0, math.sqrt(2. / (num_constraints*48)))
        bn_init(self.data_bn_p, 1)
        bn_init(self.data_bn_m, 1)

    def forward(self, x_p, x_m):
        N, C_p, T, V, M = x_p.size()
        N, C_m, T, V, M = x_m.size()

        x_p = x_p.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C_p, T)
        x_m = x_m.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C_m, T)

        x_p = self.data_bn_p(x_p)
        x_m = self.data_bn_m(x_m)

        x_p = x_p.view(N, M, V, C_p, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C_p, T, V)
        x_m = x_m.view(N, M, V, C_m, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C_m, T, V)

        # Pass through MAAF
        x_p = self.maaf_p(x_p)
        x_m = self.maaf_m(x_m)

        # Continue through TCN-GCN layers
        x_p = self.l1_p(x_p)
        x_m = self.l1_m(x_m)
        x_p = self.l2_p(x_p)
        x_m = self.l2_m(x_m)
        x_p = self.l3_p(x_p)
        x_m = self.l3_m(x_m)
        x_p = self.l4_p(x_p)
        x_m = self.l4_m(x_m)

        x_p, x_m = self.fusion1(x_p, x_m)

        x_p = self.l5_p(x_p)
        x_m = self.l5_m(x_m)
        x_p = self.l6_p(x_p)
        x_m = self.l6_m(x_m)
        x_p = self.l7_p(x_p)
        x_m = self.l7_m(x_m)

        x_p, x_m = self.fusion2(x_p, x_m)

        x_p = self.l8_p(x_p)
        x_m = self.l8_m(x_m)
        x_p = self.l9_p(x_p)
        x_m = self.l9_m(x_m)
        x_p = self.l10_p(x_p)
        x_m = self.l10_m(x_m)

        x_p, x_m = self.fusion3(x_p, x_m)

        # BGA module processing
        x_combined = self.bga(x_p)

        # N*M,C,T,V
        c_new_m = x_m.size(1)
        x_m = x_m.view(N, M, c_new_m, -1)
        x_m = x_m.mean(3).mean(1)

        c_new_p = x_p.size(1)
        x_p = x_p.view(N, M, c_new_p, -1)
        x_p = x_p.mean(3).mean(1)

        #  # Apply dropout before fully connected layers
        # x_p = self.dropout(x_p)
        # x_m = self.dropout(x_m)

        return self.fc1_classifier_p(x_p), self.fc2_aff(x_p), self.fc1_classifier_m(x_m)


class Model_Single(nn.Module):
    def __init__(self, num_class=4, num_point=16, num_person=1, graph=None, graph_args=dict(), in_channels_p=3, in_channels_m=8):
        super(Model_Single, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A

        self.data_bn = nn.BatchNorm1d(num_person * (in_channels_m+in_channels_p) * num_point)
        self.l1 = TCN_GCN_unit((in_channels_m+in_channels_p), 64, A, residual=False)

        self.l2 = TCN_GCN_unit(64, 64, A)
        self.l3 = TCN_GCN_unit(64, 64, A)
        self.l4 = TCN_GCN_unit(64, 64, A)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2)
        self.l6 = TCN_GCN_unit(128, 128, A)
        self.l7 = TCN_GCN_unit(128, 128, A)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2)
        self.l9 = TCN_GCN_unit(256, 256, A)
        self.l10 = TCN_GCN_unit(256, 256, A)

        self.fc1_classifier = nn.Linear(256, num_class)
        self.fc2_aff = nn.Linear(256, 31*48)

        nn.init.normal_(self.fc1_classifier.weight, 0, math.sqrt(2. / num_class))
        nn.init.normal_(self.fc2_aff.weight, 0, math.sqrt(2. / (31*48)))
        bn_init(self.data_bn, 1)

    def forward(self, x_p,x_m):
        N, C_p, T, V, M = x_p.size()
        N, C_m, T, V, M = x_m.size()

        x = torch.cat((x_p, x_m), 1)

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * (C_m+C_p), T)

        x = self.data_bn(x)

        x = x.view(N, M, V, (C_m+C_p), T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, (C_p+C_m), T, V)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)

        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)

        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        c_new_x = x.size(1)
        x = x.view(N, M, c_new_x, -1)
        x = x.mean(3).mean(1)

        return self.fc1_classifier(x), self.fc2_aff(x)

