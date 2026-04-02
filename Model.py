import torch
from torch import nn
import torch.nn.functional as F
from Params import args


class GCNLayer(nn.Module):
    def forward(self, adj, embeds):
        return torch.spmm(adj, embeds)


class TemporalLSTM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lstm = nn.LSTM(dim, dim, batch_first=True)

    def forward(self, seq_emb):
        _, (hn, _) = self.lstm(seq_emb)
        return hn[-1]


class AttentionFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim * 2, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, spatial, temporal):
        combined = torch.cat([spatial, temporal], dim=1)
        weights = self.softmax(self.attn(combined))

        w_s = weights[:, 0].unsqueeze(1)
        w_t = weights[:, 1].unsqueeze(1)

        return w_s * spatial + w_t * temporal


def contrastive_loss(z1, z2, temp):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    pos = torch.sum(z1 * z2, dim=1) / temp
    return -torch.mean(torch.log(torch.sigmoid(pos) + 1e-8))


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.user_emb = nn.Parameter(torch.randn(args.user, args.latdim))
        self.item_emb = nn.Parameter(torch.randn(args.item, args.latdim))

        self.gcnLayers = nn.Sequential(
            *[GCNLayer() for _ in range(args.gnn_layer)]
        )

        self.lstm = TemporalLSTM(args.latdim)
        self.fusion = AttentionFusion(args.latdim)

        self.dropout = nn.Dropout(0.2)

        nn.init.xavier_uniform_(self.user_emb)
        nn.init.xavier_uniform_(self.item_emb)

    def forward_gcn(self, adj):
        x = torch.cat([self.user_emb, self.item_emb], dim=0)

        embeds = [x]
        for gcn in self.gcnLayers:
            x = gcn(adj, x)
            x = self.dropout(x)
            embeds.append(x)

        final = sum(embeds)

        user_emb, item_emb = torch.split(final, [args.user, args.item], dim=0)

        user_emb = F.normalize(user_emb, dim=1)
        item_emb = F.normalize(item_emb, dim=1)

        return user_emb, item_emb

    # 🔥 FIXED FORWARD
    def forward(self, adj, user_seq_map, users=None):
        user_emb, item_emb = self.forward_gcn(adj)

        if users is not None:
            batch_seq = user_seq_map[users]
            seq_emb = item_emb[batch_seq]
            temporal = self.lstm(seq_emb)

            fused_users = self.fusion(user_emb[users], temporal)
            return fused_users, item_emb

        else:
            return user_emb, item_emb

    def bpr_loss(self, users, pos, neg, adj, user_seq_map):
        user_emb_batch, item_emb = self.forward(adj, user_seq_map, users)

        u = user_emb_batch
        p = item_emb[pos]
        n = item_emb[neg]

        return -torch.mean(
            F.logsigmoid(torch.sum(u * p, dim=1) - torch.sum(u * n, dim=1))
        )

    def total_loss(self, users, pos, neg, adj, user_seq_map):
        bpr = self.bpr_loss(users, pos, neg, adj, user_seq_map)

        z1, _ = self.forward(adj, user_seq_map, users)
        z2, _ = self.forward(adj, user_seq_map, users)

        cl = contrastive_loss(z1, z2, args.temp)

        return bpr + args.ssl_reg * cl