import torch




@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2*pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)




@torch.jit.script
def radial_basis(x, mean, std):
    return torch.exp((x - mean)**2 / (2*std**2 + 1e-2))
