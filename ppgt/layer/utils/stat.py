import torch






def stat_value(x, dim=None, quantile=[0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99], return_value=False):
    out = dict()

    out['min'] = x.min(dim=dim) if dim is not None else x.min().item()
    quantile.sort()
    for q in quantile:
        out[q] = torch.quantile(x, q=q, dim=dim) if dim is not None else torch.quantile(x, q=q).item()

    out['max'] = x.max(dim=dim) if dim is not None else x.max().item()

    for k in out.keys():
        print(f'{k}: {out[k]}')

    return out if return_value else None