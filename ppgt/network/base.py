import math
import torch
from torch import nn
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import new_layer_config, BatchNorm1dNode
from torch_geometric.graphgym.register import register_network


from ..layer.utils.norm import Batch2BatchNormalizationLayer


from .utils import PosEncoder
from .initialization import init_weights_vit_timm, init_weights_vit_lecun


def compute_dropout_rate(epoch, max_epoch, initial, final, mode='cosine'):
    """Compute decayed dropout rate for a given epoch.

    Supports 'cosine' and 'linear' schedules that anneal from *initial*
    down to *final* over the course of training.
    """
    if max_epoch <= 1:
        return final
    t = min(epoch / (max_epoch - 1), 1.0)
    if mode == 'cosine':
        return final + 0.5 * (initial - final) * (1 + math.cos(math.pi * t))
    elif mode == 'linear':
        return initial + (final - initial) * t
    else:
        raise ValueError(f"Unknown dropout decay mode: {mode}")


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _timm_drop_path_cls():
    try:
        from timm.layers import DropPath
        return DropPath
    except ImportError:
        from timm.models.layers import DropPath
        return DropPath



def apply_dropout_decay(
    model,
    decay_factor,
    floor=0.0,
):
    """
    Multiply every dropout / drop-path probability by ``decay_factor``, then clamp.

    Covers ``nn.Dropout`` (incl. 2d/3d), ``GraphDropPath``, and timm ``DropPath``.
    Preserves relative layerwise drop-path strengths if they differ at init.
    """
    if decay_factor <= 0:
        raise ValueError(f"decay_factor must be positive, got {decay_factor}")
    if decay_factor == 1.0:
        return

    model = _unwrap_model(model)
    dp_floor = float(floor)

    from ..layer.utils.drop_path import GraphDropPath

    TimmDropPath = _timm_drop_path_cls()

    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            new_p = float(m.p) * decay_factor
            m.p = max(floor, min(1.0, new_p))
        elif isinstance(m, GraphDropPath):
            new_dp = float(m.drop_prob) * decay_factor
            m.drop_prob = max(dp_floor, min(1.0, new_dp))
        elif isinstance(m, TimmDropPath):
            new_dp = float(m.drop_prob) * decay_factor
            m.drop_prob = max(dp_floor, min(1.0, new_dp))


class FeatureEncoder(torch.nn.Module):
    """
    Feature encoder from GraphGPS --> for feature/pos encoder defined in GraphGPS
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, dim_in, dim_edge=None):
        super(FeatureEncoder, self).__init__()
        if dim_edge is None: dim_edge = dim_in
        self.dim_in = dim_in
        self.dim_edge = dim_edge

        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = register.node_encoder_dict[
                cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(dim_in)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(dim_in, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))
            # Update dim_in to reflect the new dimension fo the node features

        if cfg.dataset.edge_encoder:
            # Hard-limit max edge dim for PNA.
            if cfg.gnn.get('dim_edge') is None:
                if 'PNA' in cfg.gt.layer_type:
                    cfg.gnn.dim_edge = min(128, dim_edge)
                else:
                    cfg.gnn.dim_edge = dim_edge

            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = register.edge_encoder_dict[
                cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(dim_edge)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(dim_edge, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch




@register_network('BaseGraphNet')
class GraphNet(torch.nn.Module):
    """PPGT network.

    Feature/positional encoders -> a stack of ``cfg.gt.layer_type`` blocks
    (``PPGTLayer`` for PPGT) -> optional class-attention layers -> prediction
    head. The block type is looked up in GraphGym's layer registry, so the same
    network definition is reused for the ablation layers.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()

        cfg.gnn.dim_inner = cfg.gt.dim_hidden
        cfg.gnn.dim_edge = cfg.gt.get('dim_edge', None)


        dim_in = cfg.gnn.dim_inner
        dim_edge = cfg.gnn.dim_edge
        self.feat_enc = FeatureEncoder(dim_in, dim_edge)
        self.pos_enc = PosEncoder(dim_in, dim_edge)

        # -----  pre-backbone normalization -----
        self.pre_backbone_norm = Batch2BatchNormalizationLayer(norm_name = cfg.gt.get('pre_backbone_norm', 'none'),
                                                    dim=cfg.gt.dim_hidden)

        # ------------ pre-backbone MPNNs (not in-use) ---------

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        # assert cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in, \
            # "The inner and hidden dims must match."

        # ----------- Backbone ---------
        layer_type = cfg.gt.get('layer_type', "CKGraphConvMLP")
        # global_model_type = "GritTransformer"

        backbone_block = register.layer_dict.get(layer_type)
        # kernel_size = cfg.gt.get('kernel_size', -1)
        # dilation = cfg.gt.get('dilation', 1)

        drop_path_rate = cfg.gt.get('drop_path', 0.)
        layerwise_drop_path_rate = cfg.gt.get('layerwise_drop_path_rate', False)
        # linear-scaling drop-rate is utilize in Stochatic Depth
        # but in CaiT, researchers find no difference from uniform drop-rate (even less stable)
        num_layers = cfg.gt.layers
        drop_path_rate = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)] if layerwise_drop_path_rate else [drop_path_rate] * num_layers
        layers = []

        if 'act' not in cfg.gt:
            cfg.gt.act = cfg.gnn.act

        for l in range(cfg.gt.layers):
            layers.append(backbone_block(
                in_dim=cfg.gt.dim_hidden,
                out_dim=cfg.gt.dim_hidden,
                num_heads=cfg.gt.n_heads,
                dropout=cfg.gt.dropout,
                attn_dropout=cfg.gt.attn_dropout,
                drop_path=drop_path_rate[l],
                act=cfg.gt.act,
                layer_norm=cfg.gt.layer_norm,
                batch_norm=cfg.gt.batch_norm,
                norm_fn=cfg.gt.norm_fn,
                residual=True,
                norm_e=cfg.gt.attn.norm_e,
                O_e=cfg.gt.attn.O_e,
                layer_idx=l,
                cfg=cfg.gt,
                # log_attn_weights=cfg.train.mode == 'log-attn-weights',
            ))

        self.layers = torch.nn.Sequential(*layers)





        # if cfg.gt.get('post_edge_ffn', False):
        #     self.post_edge_ffn = EdgeFFN(cfg.gt.dim_hidden,
        #                                  cfg.gt.dim_hidden,
        #                                  ffn_ratio=2,
        #                                  act=cfg.gnn.act,)

        # ---------- pre-backbone normalization --------
        self.post_backbone_norm = Batch2BatchNormalizationLayer(cfg.gt.get('post_backbone_norm', 'none'), cfg.gt.dim_hidden,
                                                                attr_name='x'
                                                                )

        self.class_attn = cfg.gt.get('class_attn_layers', 0) > 0
        class_attention_layers = []
        if self.class_attn:
            if "class_attn" not in cfg.gt:
                cfg.gt.class_attn = dict()
            drop_path_rate = cfg.gt.class_attn.get('drop_path', 0.)
            layerwise_drop_path_rate = cfg.gt.class_attn.get('layerwise_drop_path_rate', False)
            num_layers = cfg.gt.class_attn_layers
            drop_path_rate = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)] if layerwise_drop_path_rate else [drop_path_rate] * num_layers
            # self.cls_token_emb = node_encoder_dict['CLSTokenEmb'](out_dim=cfg.gt.dim_hidden)
            for l in range(cfg.gt.class_attn_layers):
                class_attention_layers.append(
                    backbone_block(
                        in_dim=cfg.gt.dim_hidden,
                        out_dim=cfg.gt.dim_hidden,
                        num_heads=cfg.gt.n_heads,
                        dropout=cfg.gt.dropout,
                        attn_dropout=cfg.gt.attn_dropout,
                        drop_path=drop_path_rate[l],
                        act=cfg.gt.act,
                        layer_norm=cfg.gt.layer_norm,
                        batch_norm=cfg.gt.batch_norm,
                        norm_fn=cfg.gt.norm_fn,
                        residual=True,
                        norm_e=cfg.gt.attn.norm_e,
                        O_e=cfg.gt.attn.O_e,
                        layer_idx=l,
                        class_attention=True,
                        cfg=cfg.gt,
                        # log_attn_weights=cfg.train.mode == 'log-attn-weights',
                    )
            )
            self.class_attention_layers = torch.nn.Sequential(*class_attention_layers)
            self.post_CA_norm = Batch2BatchNormalizationLayer(cfg.gt.get('post_backbone_norm', 'none'), cfg.gt.dim_hidden,
                                                                    attr_name='cls_token'
                                                                    )
        else:
            self.class_attention_layers = None

        # --------- Output Head ----------
        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out, L=cfg.gnn.layers_post_mp)

        lecun_init = cfg.gt.get('lecun_init', False)

        if lecun_init:
            self.apply(init_weights_vit_lecun)
        else:
            self.apply(init_weights_vit_timm)


    def forward(self, batch):


        with torch.no_grad():
            batch.batch_tensor = batch.ptr[1:] * 0. + 1.


        for module in self.children():
            batch = module(batch)

        return batch




