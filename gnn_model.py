"""Dense relational graph convolution and multilabel classification.

The small heterogeneous graph uses dense weighted, row-normalized adjacency
matrices. Each layer sums A_t @ H @ W_t over relations and adds H @ W_self.
Event inputs are static-plus-difference features, or constants in graph-only mode.
Non-event SDK/permission/API nodes have learned input embeddings. A linear
readout produces event logits for BCEWithLogitsLoss.

At inference, attach unseen event nodes using the frozen training-derived
co-occurrence structure and perform a forward pass without test labels."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RGCN(nn.Module):
    """Relational graph convolution with an event-level multilabel head.

    Parameters
    ----------
    d_feat : int
        Event feature dimension and non-event embedding dimension.
    n_nonapp : int
        Number of SDK, permission, and API nodes in the fixed vocabulary.
    n_edge_types : int
        Number of relations: SDK use, permission request, API occurrence,
        SDK co-occurrence, SDK removal, and SDK addition.
    n_classes : int
        Number of target categories, six for C1--C6.
    hidden : int
        Hidden representation dimension.
    n_layers : int
        Number of message-passing layers, two by default.
    dropout : float
        Hidden-layer dropout probability."""

    def __init__(self, d_feat, n_nonapp, n_edge_types, n_classes,
                 hidden=64, n_layers=2, dropout=0.5):
        super().__init__()
        self.n_edge_types = n_edge_types
        self.dropout = dropout
        # Non-event vocabularies are fixed, but embeddings are initialized and fitted per fold.
        # Never reuse embeddings learned in another fold.
        self.embed = nn.Embedding(n_nonapp, d_feat)

        # One transformation per relation plus the self transformation at index n_edge_types.
        # Use ModuleList/ParameterList; ModuleDict does not accept bare nn.Parameter entries.
        self.layer_Ws = nn.ModuleList()
        self.layer_bias = nn.ParameterList()
        d_inp = d_feat
        for _ in range(n_layers):
            self.layer_Ws.append(nn.ParameterList([
                nn.Parameter(torch.randn(d_inp, hidden) * 0.08)
                for _ in range(n_edge_types + 1)]))
            self.layer_bias.append(nn.Parameter(torch.zeros(hidden)))
            d_inp = hidden
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, feat_app, adj, app_node_ids):
        """Compute logits for the requested event nodes.

        feat_app: (n_app, d_feat) features in app_rows order.
        adj: one (n_nodes, n_nodes) row-normalized weighted adjacency tensor
             per relation; n_nodes=n_app+n_nonapp. Missing relations use zeros.
        app_node_ids: LongTensor of event IDs to read out."""
        H = torch.cat([feat_app, self.embed.weight], dim=0)  # (n_nodes, d_feat)
        for l in range(len(self.layer_Ws)):
            W = self.layer_Ws[l]
            out = self.layer_bias[l].unsqueeze(0).expand(H.size(0), -1)
            for t in range(self.n_edge_types):
                out = out + adj[t] @ H @ W[t]
            out = out + H @ W[-1]  # Self transformation preserves the node's own representation.
            H = F.relu(out)
            H = F.dropout(H, p=self.dropout, training=self.training)
        return self.head(H[app_node_ids])


class MLP(nn.Module):
    """Feature-only MLP baseline with no graph input.
    Use the same classifier-head design and optimization settings as the RGCN."""

    def __init__(self, d_in, n_classes, hidden=64, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)
