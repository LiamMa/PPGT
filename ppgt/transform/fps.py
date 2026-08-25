import torch

def fps_with_dist_mat(dist_mat, num_anchor):
    """
    > farthest_point_sampling_from_distance_matrix
    - start from the average of the nodes

    Perform farthest point sampling (FPS) given a precomputed distance matrix.
    ags:
        dist_mat (torch.Tensor): Distance matrix of shape (N, N), where dist_mat[i, j] is the distance between point i and point j.
        num_samples (int): Number of points to sample.

    Returns:
        torch.Tensor: Indices of the sampled points, of shape (num_samples,).
    """
    assert (dist_mat == dist_mat.transpose(0,1)).all(), 'dist_mat shall be symmetric'

    dist2centroid = dist_mat.mean(dim=-1, keepdim=False)


    dist_mat = torch.cat([dist_mat, dist2centroid.view(1, -1)], dim=0)
    dist_mat = torch.cat([dist_mat, torch.cat([dist2centroid, torch.zeros(1)],dim=0).view(-1, 1)], dim=1)
    # last col/row stands for centroid

    N = dist_mat.shape[0]
    # sampled_mask = torch.zero(N).type(torch.bool)
    # sampled_mask[-1] = True # always select the centroid
    sampled_indices = [-1]

    # Start with a centroid point as the first sample
    distances = torch.full((N,), float('inf'), dtype=torch.float32)

    dist2anchors = dist2centroid


    # for i in range(1, num_samples):

    num_anchor = min(num_anchor, N-1)

    node_idx = torch.arange(N-1)

    run = True
    while run:
        # Update distances with the distance to the last sampled point
        # Select the farthest point
        max_dist = torch.max(dist2anchors)

        new_anchor_mask = dist2anchors == max_dist
        new_anchor_idx = node_idx[new_anchor_mask].tolist()


        sampled_indices += new_anchor_idx

        if sampled_indices >= num_anchor:
            run = False
            break

        # todo to consider all added anchors

        # update dist to anchors
        dist_to_new_anchor = dist_mat[new_anchor_mask]
        distances = torch.min(torch.cat([distances.view(1, -1), dist_to_new_anchor], dim=0), dim=0).values

    sampled_indices = torch.LongTensor(sampled_indices)

    return sampled_indices



def anchor_pe_fps(dist_mat, pe_mat,  num_anchor):
    """
    > farthest_point_sampling_from_distance_matrix
    - start from the average of the nodes

    Perform farthest point sampling (FPS) given a precomputed distance matrix.
    ags:
        dist_mat (torch.Tensor): Distance matrix of shape (N, N), where dist_mat[i, j] is the distance between point i and point j.
        num_samples (int): Number of points to sample.

    Returns:
        torch.Tensor: Indices of the sampled points, of shape (num_samples,).
    """
    if not (dist_mat == dist_mat.transpose(0,1)).all():
        dist_mat = (dist_mat + dist_mat.transpose(0, 1)) / 2

    dist2centroid = dist_mat.mean(dim=-1, keepdim=False)


    # last col/row stands for centroid

    N = dist_mat.shape[0]
    # sampled_mask = torch.zero(N).type(torch.bool)
    # sampled_mask[-1] = True # always select the centroid
    sampled_indices = []

    # Start with a centroid point as the first sample

    dist2anchors = dist2centroid


    # for i in range(1, num_samples):

    anchor_pe = [pe_mat.mean(dim=0, keepdim=False)] #
    # num_anchor = min(num_anchor, N)

    node_idx = torch.arange(N)


    for i in range(num_anchor):
        # Update distances with the distance to the last sampled point
        # Select the farthest point
        max_dist = torch.max(dist2anchors)

        new_anchor_mask = dist2anchors == max_dist
        new_anchor_idx = node_idx[new_anchor_mask].tolist()
        sampled_indices += new_anchor_idx

        # consider the average pe to all new-anchors;  to degenerade to centroid if all nodes are selected as anchors
        pe = pe_mat[new_anchor_mask, :].mean(dim=0, keepdim=False)
        anchor_pe.append(pe)

        # todo to consider all added anchors
        # update dist to anchors
        dist_to_new_anchor = dist_mat[new_anchor_mask]
        dist2anchors = torch.min(torch.cat([dist2anchors.view(1, -1), dist_to_new_anchor], dim=0), dim=0).values

    anchor_pe = torch.cat(anchor_pe, dim=-1)

    return anchor_pe

