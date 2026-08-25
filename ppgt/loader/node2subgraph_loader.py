import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

from torch_geometric.data import InMemoryDataset

from tqdm import tqdm  # Import tqdm for progress bar
from joblib import Parallel, delayed
import warnings

from ppgt.transform.scale_rrwp import add_full_rrwp



def node2subgraph(dataset, max_hops=2, num_workers=64, debug=False, max_nodes=250, pre_compute_rrwp=False):
    # For large-scale graph, convert each a single graph into a set of subgraphs

    #  > only support transductive setting for now. e.g.,
    #    - ogbn-arxiv 

    data = dataset.data

    if pre_compute_rrwp:
        data = add_full_rrwp(data, walk_length=6, attr_name_abs="rrwp", attr_name_rel="rrwp", top_k_pruning=64)

    num_nodes = data.num_nodes

    node_indices = torch.arange(num_nodes)

    # train datasets
    train_nodes = node_indices[data.train_mask]
    val_nodes = node_indices[data.val_mask]
    test_nodes = node_indices[data.test_mask]


    def process_node(idx):
        # sample k-hop subgraph (BFS search)
        #  searching cover both in-edges and out-edges for directed graphs
        #   but the sampled subgraph retains directed


        with warnings.catch_warnings():
            node_idx = idx.item()
            device = data.edge_index.device
            num_nodes = data.num_nodes

            # 1. Initialize Masks
            subset_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            subset_mask[node_idx] = True

            current_frontier = subset_mask.clone()
            current_size = 1

            # 2. Incremental Expansion
            for i in range(max_hops):
                if current_size >= max_nodes:
                    break

                # --- CHANGED SECTION START ---
                # A. Find neighbors (Bidirectional: In and Out edges)
                
                # 1. Incoming edges (neighbors -> current_frontier)
                # Check where edges END at the frontier
                in_edge_mask = current_frontier[data.edge_index[1]]
                in_neighbors = data.edge_index[0][in_edge_mask]

                # 2. Outgoing edges (current_frontier -> neighbors)
                # Check where edges START at the frontier
                out_edge_mask = current_frontier[data.edge_index[0]]
                out_neighbors = data.edge_index[1][out_edge_mask]

                # 3. Combine both sets of neighbors
                neighbors = torch.cat([in_neighbors, out_neighbors])
                # --- CHANGED SECTION END ---

                # B. Identify STRICTLY NEW nodes
                neighbor_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
                neighbor_mask[neighbors] = True

                # Remove nodes we have already visited
                new_nodes_mask = neighbor_mask & (~subset_mask)
                num_new_nodes = new_nodes_mask.sum().item()

                if num_new_nodes == 0:
                    break

                # C. Check Budget
                if current_size + num_new_nodes <= max_nodes:
                    # Case 1: Take whole layer
                    subset_mask = subset_mask | new_nodes_mask
                    current_frontier = new_nodes_mask
                    current_size += num_new_nodes
                else:
                    # Case 2: Overflow - Sample
                    remaining_slots = max_nodes - current_size
                    new_node_indices = new_nodes_mask.nonzero(as_tuple=False).view(-1)

                    perm = torch.randperm(num_new_nodes, device=device)[:remaining_slots]
                    selected_indices = new_node_indices[perm]

                    subset_mask[selected_indices] = True
                    break

            # 3. Finalize
            subset = subset_mask.nonzero(as_tuple=False).view(-1)

            # Extract subgraph
            # This preserves the original edge direction because it filters 
            # the original directed data.edge_index
            edge_index, edge_attr = subgraph(subset, data.edge_index, edge_attr=data.edge_attr, relabel_nodes=True, return_edge_mask=False)

            # 4. Build Data Object
            sub_data = Data()
            sub_data.edge_index = edge_index
            sub_data.edge_attr = edge_attr
            sub_data.num_nodes = subset.size(0)

            if "rrwp_index" in data:
                rrwp_index, rrwp_attr = subgraph(subset, data.rrwp_index, edge_attr=data.rrwp_attr, relabel_nodes=True, return_edge_mask=False)
                sub_data.rrwp_index = rrwp_index
                sub_data.rrwp_attr = rrwp_attr

            if hasattr(data, 'x') and data.x is not None:
                sub_data.x = data.x[subset]

            if hasattr(data, 'y') and data.y is not None:
                if data.y.dim() > 0:
                    sub_data.y = data.y[idx]
                else:
                    sub_data.y = data.y

            sub_data.original_idx = idx
            
            # Note: After relabeling, the target node will be at the index 
            # corresponding to its position in 'subset'.
            # If you need the specific new index of the center node:
            # sub_data.new_center_idx = (subset == node_idx).nonzero().item()
            
            sub_data.target_mask = subset == node_idx

        return sub_data


    test_graphs = Parallel(n_jobs=num_workers, backend='threading')(delayed(process_node)(idx) for idx in tqdm(test_nodes, desc="converting test nodes to test graphs", unit="node"))
    test_dataset = dataset_from_data_list(test_graphs, transform=dataset.transform)

    val_graphs = Parallel(n_jobs=num_workers, backend='threading')(delayed(process_node)(idx) for idx in tqdm(val_nodes, desc="converting validation nodes to validation graphs", unit="node"))
    val_dataset = dataset_from_data_list(val_graphs, transform=dataset.transform)

    train_graphs = Parallel(n_jobs=num_workers, backend='threading')(delayed(process_node)(idx) for idx in tqdm(train_nodes, desc="converting training nodes to training graphs", unit="node"))
    train_dataset = dataset_from_data_list(train_graphs, transform=dataset.transform)

    # ------- for debugging only -------
    # train_dataset = test_dataset
    # val_dataset = test_dataset
    # test_dataset = test_dataset

    del dataset

    print("max_nodes: ", max_nodes, "max_hops: ", max_hops)
    print(np.histogram([i.x.size(0) for i in train_dataset]))
    print(np.histogram([i.x.size(0) for i in val_dataset]))
    print(np.histogram([i.x.size(0) for i in test_dataset]))


    return train_dataset, val_dataset, test_dataset


def dataset_from_data_list(data_list, transform=None):
    data_list = list(filter(None, data_list))

    dataset = InMemoryDataset(transform=transform)

    dataset._indices = None
    dataset.data_list = data_list
    dataset.data, dataset.slices = dataset.collate(data_list)
    return dataset



# def process_node(idx):
#     """
#     Formulates a subgraph extraction by searching for the largest k-hop neighborhood
#     of node idx such that the total number of nodes does not exceed max_nodes.
#     Falls back to random downsampling of 1-hop neighbors if needed.
#     """
#     with warnings.catch_warnings():
#         low, high = 1, max_hops
#         best_hops = 0
#         best_result = None

#         # Formulate: Find maximal hops (<=max_hops) s.t. subgraph size <= max_nodes
#         while low <= high:
#             cur_hops = (low + high) // 2
#             subset, _, _, _ = k_hop_subgraph(
#                 idx.item(), cur_hops, data.edge_index, relabel_nodes=False
#             )
#             if subset.size(0) > max_nodes:
#                 high = cur_hops - 1
#             else:
#                 best_hops = cur_hops
#                 low = cur_hops + 1

#         if best_hops > 0:
#             # Formulate clean subgraph with relabeled nodes
#             subset, edge_index, mapping, edge_mask = k_hop_subgraph(
#                 idx.item(), best_hops, data.edge_index, relabel_nodes=True
#             )
#         else:
#             # Fallback: 1-hop, randomly downsample if too many nodes
#             subset, _, _, _ = k_hop_subgraph(
#                 idx.item(), 1, data.edge_index, relabel_nodes=False
#             )
#             if subset.size(0) > max_nodes:
#                 perm = torch.randperm(subset.size(0), device=subset.device)[:max_nodes]
#                 subset = subset[perm]
#             center = torch.tensor([idx.item()], device=data.edge_index.device)
#             subset = torch.cat([center, subset]).unique()
#             edge_index, _ = subgraph(
#                 subset, data.edge_index, relabel_nodes=True, return_edge_mask=False
#             )
#             mapping = None
#             edge_mask = None

#         # Formulate the subgraph data object
#         sub_data = Data()
#         sub_data.edge_index = edge_index
#         sub_data.num_nodes = subset.size(0)
#         if hasattr(data, 'x') and data.x is not None:
#             sub_data.x = data.x[subset]
#         if hasattr(data, 'y') and data.y is not None:
#             if data.y.dim() > 0:
#                 sub_data.y = data.y[idx].view(1)
#             else:
#                 sub_data.y = data.y
#         sub_data.original_idx = idx
#         sub_data.target_mask = subset == idx.item()
#     return sub_data
