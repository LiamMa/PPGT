# from torch_scatter import scatter




# todo:

# @register_head('vn_graph')
# class VirtualNodeGraphHead(nn.Module):
#     """
#     With Edge and Node pooling + Num-nodes counting
#     Args:
#         dim_in (int): Input dimension.
#         dim_out (int): Output dimension. For binary prediction, dim_out=1.
#         L (int): Number of hidden layers.
#     """
#     def __init__(self, dim_in, dim_out, L=2, agg_edge=False):
#         super().__init__()
#         self.pooling_fun = register.pooling_dict[cfg.model.graph_pooling]
#
#         self.norm = nn.Identity()
#         batch_norm = cfg.gnn.get('batch_norm', False)
#         layer_norm = cfg.gnn.get('layer_norm', False)
#
#         out_bn = cfg.gnn.get('out_bn', False)
#         if out_bn:
#             self.out_bn = nn.BatchNorm1d(dim_out)
#         else:
#             self.out_bn = nn.Identity()
#
#         if layer_norm:
#             self.norm = nn.LayerNorm(dim_in)
#
#         if batch_norm:
#             self.norm = nn.BatchNorm1d(dim_in)
#
#         dim_in = dim_in * 2
#         self.g_order_coef = nn.Parameter(torch.ones(1, dim_in), requires_grad=True)
#
#         list_FC_layers = [
#             nn.Linear(dim_in // 2 ** l, dim_in // 2 ** (l + 1), bias=True)
#             for l in range(L)]
#         list_FC_layers.append(
#             nn.Linear(dim_in // 2 ** L, dim_out, bias=True))
#         self.FC_layers = nn.ModuleList(list_FC_layers)
#         self.L = L
#         self.activation = register.act_dict[cfg.gnn.act]()
#
#         # note: modified to add () in the end from original code of 'GPS'
#         #   potentially due to the change of PyG/GraphGym version
#
#     def _apply_index(self, batch):
#         return batch.graph_feature, batch.y
#
#     def forward(self, batch):
#         graph_order = batch.ptr[1:] - batch.ptr[:-1]
#         graph_emb_node = self.pooling_fun(batch.x, batch.batch)
#         graph_emb_edge = self.pooling_fun(batch.edge_attr, batch.batch[batch.edge_index[1]])
#         graph_emb = torch.cat([graph_emb_node, graph_emb_edge], dim=-1)
#
#         graph_emb = graph_emb + graph_emb * (graph_order.view(-1, 1) * self.g_order_coef)
#         graph_emb = self.norm(graph_emb)
#         for l in range(self.L):
#             graph_emb = self.FC_layers[l](graph_emb)
#             graph_emb = self.activation(graph_emb)
#         graph_emb = self.FC_layers[self.L](graph_emb)
#         batch.graph_feature = graph_emb
#         pred, label = self._apply_index(batch)
#         return self.out_bn(pred), label
#         # out_bn is for BREC only; disable for others
#
#
#
#
