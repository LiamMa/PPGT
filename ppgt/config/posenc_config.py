from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('posenc')
def set_cfg_posenc(cfg):
    """Extend configuration with positional encoding options.
    """

    # Argument group for each Positional Encoding class.
    cfg.posenc_LapPE = CN()
    cfg.posenc_SignNet = CN()
    cfg.posenc_RWSE = CN()
    cfg.posenc_HKdiagSE = CN()
    cfg.posenc_ElstaticSE = CN()
    cfg.posenc_EquivStableLapPE = CN()
    cfg.posenc_RRWP = CN()
    cfg.posenc_RRWF = CN()
    cfg.posenc_ChebII = CN()
    cfg.posenc_addX= CN()
    cfg.posenc_JRRWP = CN()
    cfg.posenc_PathWalk = CN()
    cfg.posenc_PathCount= CN()
    cfg.posenc_SubCount = CN()
    cfg.posenc_HomCount = CN()
    cfg.posenc_ESC = CN()
    cfg.posenc_AddVN = CN()
    cfg.posenc_I2SG = CN()
    cfg.posenc_I2SE = CN()
    cfg.posenc_EgoSG = CN()
    cfg.posenc_CycleCount= CN()
    cfg.posenc_KP = CN()
    cfg.posenc_HyperGraph= CN()
    cfg.posenc_SPE = CN()
    cfg.posenc_RingCount= CN()
    cfg.posenc_AddCLS= CN()



    # Common arguments to all PE types.
    for name in ['posenc_LapPE', 'posenc_SignNet',
                 'posenc_RWSE', 'posenc_HKdiagSE', 'posenc_ElstaticSE',
                 'posenc_RRWP', 'posenc_RRWF', 'posenc_ChebII',
                 'posenc_addX', 'posenc_JRRWP', 'posenc_PathWalk',
                 'posenc_PathCount', 'posenc_SubCount', 'posenc_HomCount',
                 'posenc_ESC', 'posenc_AddVN',
                 'posenc_I2SG', 'posenc_EgoSG', 'posenc_I2SE',
                 'posenc_CycleCount',
                 'posenc_KP',
                 'posenc_HyperGraph',
                 'posenc_SPE',
                 'posenc_RingCount',
                 'posenc_AddCLS'
                 ]:
        pecfg = getattr(cfg, name)
        # Use extended positional encodings
        pecfg.enable = False

        # Neural-net model type within the PE encoder:
        # 'DeepSet', 'Transformer', 'Linear', 'none', ...
        pecfg.model = 'none'

        # Size of Positional Encoding embedding
        pecfg.dim_pe = 16

        # Number of layers in PE encoder model
        pecfg.layers = 3

        # Number of attention heads in PE encoder when model == 'Transformer'
        pecfg.n_heads = 4

        # Number of layers to apply in LapPE encoder post its pooling stage
        pecfg.post_layers = 0

        # Choice of normalization applied to raw PE stats: 'none', 'BatchNorm'
        pecfg.raw_norm_type = 'none'

        # In addition to appending PE to the node features, pass them also as
        # a separate variable in the PyG graph batch object.
        pecfg.pass_as_var = False

    # Config for EquivStable LapPE
    cfg.posenc_EquivStableLapPE.enable = False
    cfg.posenc_EquivStableLapPE.raw_norm_type = 'none'

    # Config for Laplacian Eigen-decomposition for PEs that use it.
    for name in ['posenc_LapPE', 'posenc_SignNet', 'posenc_EquivStableLapPE']:
        pecfg = getattr(cfg, name)
        pecfg.eigen = CN()

        # The normalization scheme for the graph Laplacian: 'none', 'sym', or 'rw'
        pecfg.eigen.laplacian_norm = 'sym'

        # The normalization scheme for the eigen vectors of the Laplacian
        pecfg.eigen.eigvec_norm = 'L2'

        # Maximum number of top smallest frequencies & eigenvectors to use
        pecfg.eigen.max_freqs = 10

    # Config for SignNet-specific options.
    cfg.posenc_SignNet.phi_out_dim = 4
    cfg.posenc_SignNet.phi_hidden_dim = 64

    for name in ['posenc_RWSE', 'posenc_HKdiagSE', 'posenc_ElstaticSE']:
        pecfg = getattr(cfg, name)

        # Config for Kernel-based PE specific options.
        pecfg.kernel = CN()

        # List of times to compute the heat kernel for (the time is equivalent to
        # the variance of the kernel) / the number of steps for random walk kernel
        # Can be overridden by `posenc.kernel.times_func`
        pecfg.kernel.times = []

        # Python snippet to generate `posenc.kernel.times`, e.g. 'range(1, 17)'
        # If set, it will be executed via `eval()` and override posenc.kernel.times
        pecfg.kernel.times_func = ''

    # Override default, electrostatic kernel has fixed set of 10 measures.
    cfg.posenc_ElstaticSE.kernel.times_func = 'range(10)'

    # ----------------- Note: FullRRWP --------------
    cfg.posenc_RRWP.enable = False
    cfg.posenc_RRWP.ksteps = 21
    cfg.posenc_RRWP.add_identity = True
    cfg.posenc_RRWP.add_uniform = False
    cfg.posenc_RRWP.add_node_attr = False
    cfg.posenc_RRWP.add_inverse = False
    cfg.posenc_RRWP.spd = False
    cfg.posenc_RRWP.topk = None
    cfg.posenc_RRWP.add_flow = False
    cfg.posenc_RRWP.add_attr = False
    cfg.posenc_RRWP.orthogonal = False
    cfg.posenc_RRWP.kp_encoding = False
    cfg.posenc_RRWP.kp_order = 4

    # ----------------- Note: Jump RRWP --------------
    cfg.posenc_JRRWP.enable = False
    cfg.posenc_JRRWP.ksteps = 10
    cfg.posenc_JRRWP.jump_step = 1

    # ---------------- Note: Flow-based RRWP ---------
    cfg.posenc_RRWF.enable = False
    cfg.posenc_RRWF.ksteps = 21
    cfg.posenc_RRWF.add_identity = True
    cfg.posenc_RRWF.spd = False
    cfg.posenc_RRWF.timesN = False
    cfg.posenc_RRWF.scale_factor = 1.

    # ----------------- Note: Cheb-Basis II --------------
    cfg.posenc_ChebII.enable = False
    cfg.posenc_ChebII.ksteps = 21
    cfg.posenc_ChebII.to_rrwp = False # convert to Random walk like

    # ----------------- add dummy-X for graph without x --------------
    cfg.posenc_addX.enable = False


    # ----------------- Note: PathWalk --------------
    cfg.posenc_PathWalk.enable = False
    cfg.posenc_PathWalk.ksteps = 21
    cfg.posenc_PathWalk.add_identity = True
    cfg.posenc_PathWalk.topk = None

    # ----------------- Note: PathCount--------------
    cfg.posenc_PathCount.enable = False
    cfg.posenc_PathCount.ksteps = 21
    cfg.posenc_PathCount.add_identity = True
    cfg.posenc_PathCount.topk = None


    # ----------------- Note: SubGraph Count --------------
    cfg.posenc_SubCount.enable = False
    cfg.posenc_SubCount.id_scopre = 'local' # 'local' for gsn-e; 'global' for gsn-v
    cfg.posenc_SubCount.k = 3
    cfg.posenc_SubCount.id_type = 'all_simple_graphs'  # or ['cycle_graph']

    cfg.posenc_SubCount.induced = False
    cfg.posenc_SubCount.directed = False
    cfg.posenc_SubCount.directed_orbits = False
    cfg.posenc_SubCount.edge_automorphism = 'induced'

    cfg.posenc_SubCount.custom_edge_list = None

    cfg.posenc_SubCount.loaded = False  # to indiciate bypassing PE computation for loading preprocessed SubCount





    # ----------------- Note: Homomorphism Count --------------
    cfg.posenc_HomCount.enable = False
    cfg.posenc_HomCount.file = 'all_simple_graphs'



    # ----------------- Note: ESC (efficient subgraph PE) --------------
    cfg.posenc_ESC.enable = False

    # ----------------- Note: VN --------------
    cfg.posenc_AddVN.enable = False
    cfg.posenc_AddVN.num_vn = 1


    # ----------------- Note: I2 SubGraph --------------
    cfg.posenc_I2SG.enable = False
    cfg.posenc_I2SG.k_hop = 4
    cfg.posenc_I2SG.pe_name = 'rrwp'
    cfg.posenc_I2SG.add_self_loops=False


    # ----------------- Note: Ego SubGraph --------------
    cfg.posenc_EgoSG.enable = False
    cfg.posenc_EgoSG.k_hop = 4


    # ----------------- Note: Cycle Count --------------
    cfg.posenc_CycleCount.enable = False
    cfg.posenc_CycleCount.k = 16

    # ----------------- Note: KP edges ----------------
    ## convert egdges to KP (K-hop per
    cfg.posenc_KP.enable = False
    cfg.posenc_KP.k_hop = 4
    cfg.posenc_KP.pe_name = 'rrwp'

    # ----------------- Note: HyperGraph -----------   ??? not sure about the naming
    cfg.posenc_HyperGraph.enable = False

    # ----------------- Note: SPE -----
    cfg.posenc_SPE.enable = False


    # ------------
    cfg.posenc_RingCount.enable = False


    # ------- Note: add class token
    cfg.posenc_AddCLS.enable = False
