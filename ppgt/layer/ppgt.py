import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg

# from .ckgconv_layer_ffn import xavier_normal_
# from torch_scatter import scatter, scatter_max, scatter_add


from torch_geometric.graphgym.register import *
from einops import einsum, rearrange



# from .utils.scatter import softmax as pyg_softmax

from .utils.residual import ResidualLayer
from .utils.mlp import FFNSwiGLU
from .utils.drop_path import GraphDropPath # , add_batch_tensor # the GraphDropPath is too slow --> 3 ~ 5 x time

from .utils.norm import NormalizationLayer





class URPEL2AttentionFast(nn.Module):
    """Attention module of PPGT.

    Multi-head attention over the complete graph with

    - a learned relative positional bias ``P_r`` computed from the node-pair
      encoding, and, optionally,
    - the ``l2dist`` term that turns the dot product into a (negated) squared
      distance, and
    - URPE (``P_d``), a multiplicative per-pair gate applied after the softmax.

    Follows the attention layout of GRIT (Ma et al., ICML 2023).
    """
    def __init__(self, in_dim, out_dim, num_heads,
                 dim_edge=None,
                 clamp=5., dropout=0.,
                 urpe=True,
                 l2dist=True,
                 layer_idx=-1,
                 class_attention=False, # class attention layer in CaiT
                 learn_k_bias=False,
                 cfg=dict(),
                 **kwargs
                 ):
        super().__init__()


        self.gated_attn = cfg.get('gated_attn', False)
        if self.gated_attn:
            self.G = nn.Linear(in_dim, out_dim * num_heads)


        self.out_dim = out_dim
        self.num_heads = num_heads
        self.attn_dropout = nn.Dropout(dropout)
        self.clamp = np.abs(clamp) if clamp is not None else None
        self.urpe = urpe
        self.layer_idx = layer_idx

        self.lecun_init = cfg.get('lecun_init', False)
        self.xavier_init = cfg.get('xavier_init', False)

        self.class_attention = class_attention
        if self.class_attention:
            self.urpe = False

        self.include_cls = cfg.get('include_cls', False)
        # do not include cls-token in the self-attention (inspired by CaiT)


        # class-attention layer proposed in CaiT
        #   - cross-attention layer from nodes to cls-tokens

        # scaled dot product
        qkv_bias = cfg.get('qkv_bias', True)
        o_bias = cfg.get('o_bias', True)

        self.window_attn_hop = cfg.get('window_attn_hop', -1)

        # self.QKV = nn.Linear(in_dim, 3 * out_dim * num_heads, bias=qkv_bias)
        self.Q = nn.Linear(in_dim, 1 * out_dim * num_heads, bias=qkv_bias)
        self.KV = nn.Linear(in_dim, 2 * out_dim * num_heads, bias=qkv_bias)
        # self.V = nn.Linear(in_dim, 1 * out_dim * num_heads, bias=qkv_bias)
        self.O = nn.Linear(out_dim * num_heads, in_dim, bias=o_bias)

        self.xN = cfg.get('xN', False)


        self.learn_k_bias = learn_k_bias
        if learn_k_bias:
            self.k_sigma = nn.Parameter(torch.ones(num_heads), requires_grad=True)


        if dim_edge is None: dim_edge = in_dim
        self.P = nn.Linear(dim_edge, num_heads if not self.urpe else num_heads * 2, bias=True)


        if self.class_attention:
            # remove RPE for class_attention
            self.P = nn.Identity()


        # self.l2dist = cfg.get('l2dist', True)  # for debugging only
        self.l2dist = l2dist


        self.zero_token = cfg.get('zero_token', False)
        if self.zero_token:
            self.Z = nn.Linear(in_dim, num_heads)


        qk_norm = cfg.get('qk_norm', 'none')
        self.q_norm = act_dict[qk_norm](in_dim)
        self.k_norm = act_dict[qk_norm](in_dim)



        # PE_scale = cfg.get('PE_scale', 1.0)

        # if PE_scale != 1.0:
        #     self.PE_scale = True
        #     self.PE_scale_factor = nn.Parameter(torch.ones(1, self.num_heads, 1, 1) * PE_scale, requires_grad=True)
        # else:
        #     self.PE_scale = False






    def forward(self, q, kv, batch):

        max_num_nodes= torch.max(batch.ptr[1:] - batch.ptr[:-1])
        hid_dim = self.out_dim * self.num_heads

        q = self.Q(q)
        kv = self.KV(kv)

        with torch.no_grad():
            if 'KV' not in batch:
                KV, kv_mask = pyg.utils.to_dense_batch(kv*0, batch.batch)
                batch.KV = KV 
                batch.kv_mask = kv_mask
            else:
                KV = batch.KV.detach()
                kv_mask = batch.kv_mask

        KV = KV * 0
        KV[kv_mask] = kv # (B, N)
        K, V = KV[:, :, :hid_dim], KV[:, :, hid_dim:]
        K = rearrange(self.k_norm(K), 'b m (h d) -> b h m d', h=self.num_heads)
        V = rearrange(V, 'b m (h d) -> b h m d', h=self.num_heads)

        # ---- Pre-process Q and RPE ----
        Q, q_mask, P_r, P_d = self._self_attn(q, batch) if not self.class_attention else self._class_attn(q, batch)

        Q = rearrange(self.q_norm(Q), 'b n (h d) -> b h n d', h=self.num_heads)

        if self.l2dist:
            K_norm = -0.5 * einsum(K, K, 'b h m d, b h m d -> b h m') / np.sqrt(self.out_dim)
            K_norm = rearrange(K_norm, 'b h (n m) -> b h n m', n=1)
            if self.learn_k_bias:
                K_norm = K_norm * self.k_sigma.view(1, -1, 1, 1)

            attn_bias = P_r + K_norm
        else:
            attn_bias = P_r

        kv_mask = rearrange(kv_mask, 'b (h n m) -> b h n m', n=1, h=1)

        if "swa_mask" in batch:
            kv_mask = kv_mask & batch.swa_mask
        

        # if self.PE_scale:
        #     attn_bias = attn_bias * self.PE_scale_factor

        attn = einsum(Q, K, 'b h n d, b h m d -> b h n m') + attn_bias
        attn = attn.masked_fill(~kv_mask, -1e16)

        # Attention with URPE if P_d is not 1.
        attn = F.softmax(attn, dim=-1) * P_d
        attn = self.attn_dropout(attn)

        O = einsum(attn, V, 'b h n m, b h m d -> b h n d')
        O = rearrange(O, 'b h n d -> b n (h d)')

        o = O[q_mask] if q_mask is not None else O.squeeze(1)

        if self.gated_attn:
            g = F.sigmoid(self.G(q))
            o = o * g

        return self.O(o) # only return the true node


    def _self_attn(self, q, batch):
        max_num_nodes= torch.max(batch.ptr[1:] - batch.ptr[:-1])
        with torch.no_grad():
            if 'Q' not in batch:
                Q, q_mask = pyg.utils.to_dense_batch(q*0, batch.batch)
                batch.Q = Q
                batch.q_mask = q_mask
            else:
                Q = batch.Q.detach() 
                q_mask = batch.q_mask
        Q = Q * 0
        Q[q_mask] = q


        if self.window_attn_hop > 0 and "swa_mask" not in batch:
        #     '''only supporting RRWP as PE-index; to do with spd'''

            # the first column in rrwp is the self-identification
            rrwp_mask = batch.rrwp_attr[:, :self.window_attn_hop+2].sum(dim=-1) > 0
            swa_index = batch.rrwp_index[:, rrwp_mask]

            swa_mask = pyg.utils.to_dense_adj(swa_index, batch.batch,
                                       edge_attr=None,
                                       max_num_nodes=max_num_nodes
                                       ).transpose(1, 2) # original GRIT using right matmul --> transpose for left matmul
            batch.swa_mask = (swa_mask > 0).unsqueeze(1)


        if 'PE' in batch:
            P = batch.PE
        else:
            P = pyg.utils.to_dense_adj(batch.edge_index, batch.batch,
                                       edge_attr=batch.edge_attr,
                                       max_num_nodes=max_num_nodes
                                       ).transpose(1, 2) # original GRIT using right matmul --> transpose for left matmul
            batch.PE = P

        P = self.P(P)
        hid_dim = self.out_dim * self.num_heads

        P_r = P[..., :self.num_heads]
        P_d = (P[..., self.num_heads:]+1) if self.urpe else 1.
        P_r, P_d = rearrange(P_r, 'b n m h -> b h n m'), rearrange(P_d, 'b n m h -> b h n m')
        # (B, N, M, H)

        return Q, q_mask, P_r, P_d


    def _class_attn(self, q, batch):
        # no URPE and RPE in attn
        max_num_nodes= torch.max(batch.ptr[1:] - batch.ptr[:-1])
        Q = q.unsqueeze(1) # (B, D) --> (B, 1, D)
        P_r, P_d = 0, 1.
        q_mask = None
        return Q, q_mask, P_r, P_d



    def __repr__(self):
        return f'{super().__repr__()}(l2dist={self.l2dist}, urpe={self.urpe}, head={self.num_heads}, learn_k_bias={self.learn_k_bias})'


    #


@register_layer("PPGTLayer")
class PPGTLayer(nn.Module):
    """
        PPGT Layer 
    """
    def __init__(self, in_dim, out_dim, num_heads,
                 dropout=0.0,
                 attn_dropout=0.0,
                 drop_path=0.,
                 # layer_norm=False, batch_norm=True,
                 norm_fn='batch_norm',
                 residual=True,
                 act='relu',
                 layer_idx=-1,
                 class_attention=False,
                 cfg=dict(),
                 **kwargs):
        super().__init__()


        self.debug = False
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.residual = residual
        # self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # note: feed drop_path rate from backbone to adaptively set by the depth
        # drop_path = cfg.get('drop_path', 0.)
        drop_path_scale = cfg.get('drop_path_scale', True)
        self.drop_path = GraphDropPath(drop_path, scale_by_keep=drop_path_scale) # auto to be identity() inside the function if drop-rate is 0

        self.ffn_drop = max(dropout, cfg.get('ffn_drop', 0.))
        proj_drop = cfg.get('proj_drop', 0.)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0. else nn.Identity()

        self.class_attention = class_attention

        # self.residual_rescale = cfg.get('residual_rescale', False)
        # ---------- Residual Config -----------
        layer_scale= cfg.get('layer_scale', False)
        rezero = cfg.get('rezero', False)
        alpha = cfg.get('alpha', 1.)
        # residual connection

        # ------- Normalization Layer Config --------

        self.macro_norm_type = cfg.get('macro_norm_type', 'pre_norm')
        self.norm_index = None

        # -------------- Attention --------------
        if cfg.get("attn", None) is None:
            cfg.attn = dict()
        self.use_attn = cfg.attn.get("use", True)
        # self.sigmoid_deg = cfg.attn.get("sigmoid_deg", False)
        # self.norm1_h = norm_fn(out_dim)

        self.norm1_h = NormalizationLayer(norm_fn, out_dim)

        self.attention = URPEL2AttentionFast(
            in_dim=in_dim,
            out_dim=out_dim // num_heads,
            num_heads=num_heads,
            dim_edge=cfg.get('dim_edge'),
            dropout=attn_dropout,
            clamp=cfg.attn.get("clamp", None),
            urpe= cfg.attn.get('urpe', True),
            l2dist=cfg.attn.get('l2dist', True),
            # zero_token=cfg.attn.get('zero_token', False),
            layer_idx=layer_idx,
            class_attention=class_attention,
            norm_fn=norm_fn,
            norm_P=cfg.attn.get('norm_P', False),
            learn_k_bias=cfg.attn.get('learn_k_bias', False),
            cfg=cfg.attn,
            **kwargs
        )

        # -------- Deg Scaler Option ------
        # self.deg_scaler = cfg.attn.get("deg_scaler", False)
        # if self.deg_scaler:
        #     self.deg_coef = nn.Parameter(torch.zeros(2, 1, out_dim//num_heads * num_heads))
        #     nn.init.xavier_normal_(self.deg_coef)

        self.res_layer1 = ResidualLayer(rezero=rezero, layer_scale=layer_scale, alpha=alpha, dim=out_dim) if self.residual else lambda x,y: x

        # ----------- FFN ---------------
        self.norm2_h = NormalizationLayer(norm_fn, out_dim)
        self.ffnbn = cfg.get('ffnbn', False)
        # self.act = act_dict[act]() if act is not None else nn.Identity()
        # self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        # self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)
        ffn_ratio = cfg.get('ffn_ratio', 2)

        if act.lower() == 'swiglu':
            self.FFN = FFNSwiGLU(out_dim, out_dim , hid_dim=int(out_dim * ffn_ratio * 2/3))
        else:
            self.FFN = nn.Sequential(
                nn.Linear(out_dim, int(out_dim * ffn_ratio)),
                nn.Identity() if not self.ffnbn else norm_fn((out_dim * ffn_ratio)),
                act_dict[act](),
                # nn.Dropout(ffn_dropout) if ffn_dropout > 0 else nn.Identity(),
                nn.Dropout(self.ffn_drop) if self.ffn_drop > 0 else nn.Identity(),
                nn.Linear(int(out_dim * ffn_ratio), out_dim)
            )


        self.res_layer2 = ResidualLayer(rezero=rezero, layer_scale=layer_scale, alpha=alpha, dim=out_dim) if self.residual else lambda x,y: x

        self.out_dim = out_dim
        # ---------------  Extra Normalization Config -----------


        if self.class_attention:
            assert self.macro_norm_type == 'pre_norm', 'currently only support pre-norm for class-attention'


    def decay_dropout_rate(self, decay_factor=1.0):
        """Multiply dropout / drop-path rates by *decay_factor*, then clamp.

        Applies to attention dropout, projection dropout, FFN dropout (if present), and
        :class:`GraphDropPath`. Relative strengths are preserved within this block.
        """
        if decay_factor <= 0:
            raise ValueError(f"decay_factor must be positive, got {decay_factor}")
        if decay_factor == 1.0:
            return

        lo, hi = 0., 1.0

        def _mul_drop(m):
            if isinstance(m, nn.Dropout):
                m.p = max(lo, min(hi, float(m.p) * decay_factor))

        _mul_drop(self.attention.attn_dropout)
        _mul_drop(self.proj_drop)
        for mod in self.FFN.modules():
            _mul_drop(mod)
        self.drop_path.drop_prob = max(lo, min(hi, float(self.drop_path.drop_prob) * decay_factor))


    def forward(self, batch):

        if self.macro_norm_type == 'pre_norm':
            return self._forward_pre_norm(batch)

    def _forward_pre_norm(self, batch):
        num_nodes = batch.num_nodes
        batch_tensor = batch.get('batch_tensor', None)


        batch_index = batch.batch if not self.class_attention else None

        if not self.class_attention:
            h = h_in1 = batch.x
            h = self.norm1_h(h, batch_index)
            k = h
        else:
            h = h_in1 = batch.cls_token
            h = self.norm1_h(h, None)
            k = batch.x

        h = self.attention(h, k, batch)

        h = self.proj_drop(h)
        if self.residual:
            h = self.drop_path(h, batch_index, batch_tensor)
            h = self.res_layer1(h, h_in1)

        # FFN for h
        h_in2 = h  # for second residual connection
        h = self.norm2_h(h, batch.batch)
        h = self.FFN(h)

        h = self.proj_drop(h)
        if self.residual:
            h = self.drop_path(h, batch_index, batch_tensor)
            h = self.res_layer2(h, h_in2)


        if self.class_attention:
            batch.cls_token = h
        else:
            batch.x = h

        return batch

    #
    def __repr__(self):
        return '(in_channels={}, out_channels={}, heads={}, residual={}, norm_type={}, class_attn={})\n[{}]'.format(
            # self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.num_heads,
            self.residual,
            self.macro_norm_type,
            self.class_attention,
            super().__repr__(),
        )





