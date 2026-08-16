import math
import time

import numpy as np
import torch
from scipy.special import gamma

from fpinns.helper import smooth_ceil


def fraccaputo(yt, h, a, tau):
    n = len(yt)
    frac_order = yt.clone()
    for k in range(1, n):
        memory = 0.0
        mem = [0, 0]
        mem[0] = max(0, k - math.ceil(tau / h))
        mem[1] = k
        k_st = mem[1] - mem[0]
    w = np.zeros(k_st)
    for j in range(k_st):
        if j == k_st - 1:
            w[k_st - 1 - j] = (j + 2) ** (1 - a) - 3 * (j + 1) ** (1 - a) + 2 * (j) ** (1 - a)
        else:
            w[k_st - 1 - j] = (j + 2) ** (1 - a) - 2 * (j + 1) ** (1 - a) + (j) ** (1 - a)
        memory += yt[k - 1 - j] * w[k_st - 1 - j]
    frac_order[k] = (h ** (-a) / gamma(2 - a)) * (yt[k - 1] + memory)
    return frac_order


def fraccaputo_v2(yt, h, a, tau, k):
    n = len(yt)
    frac_order = 0.0
    memory = 0.0
    mem = [0, 0]
    mem[0] = max(0, k - math.ceil(tau / h))
    mem[1] = k
    k_st = mem[1] - mem[0]
    w = np.zeros(k_st)
    for j in range(k_st):
        if j == k_st - 1:
            w[k_st - 1 - j] = (j + 2) ** (1 - a) - 3 * (j + 1) ** (1 - a) + 2 * (j) ** (1 - a)
        else:
            w[k_st - 1 - j] = (j + 2) ** (1 - a) - 2 * (j + 1) ** (1 - a) + (j) ** (1 - a)
    memory += yt[k - 1 - j] * w[k_st - 1 - j]
    frac_order = (h ** (-a) / gamma(2 - a)) * (yt[k - 1] + memory)
    return frac_order


def fraccaputo_V3(yt, h, a, tau, k):
    device = yt.device
    st = time.time()
    frac_order = torch.zeros(yt.shape[0], 1)
    memory = torch.zeros(yt.shape[0], 1)
    mem = torch.tensor([0, 0])
    mem[0] = torch.max(torch.tensor(0), k - torch.ceil(tau / h))
    mem[1] = k
    k_st = mem[1] - mem[0]
    j = torch.arange(k_st, device=device)
    j = torch.flip(j, dims=[0])
    w = torch.zeros_like(j, dtype=yt.dtype, device=device)
    w[0] = (j[0] + 2) ** (1 - a) - 3 * (j[0] + 1) ** (1 - a) + 2 * j[0] ** (1 - a)
    w[1:-1] = (j[1:-1] + 1) ** (1 - a) - 2 * (j[1:-1]) ** (1 - a) + (j[1:-1] - 1) ** (1 - a)
    w[-1] = 1
    fracorder = (h ** (-a) / torch.exp(torch.lgamma(2 - a))) * torch.sum(yt[-k_st:] * w)
    ed = time.time()
    ittim = ed - st
    return fracorder


def fraccaputo_V4(yt, h, a, tau):
    device = yt.device
    st = time.time()
    ytlen = yt.shape[0]
    frac_order = torch.zeros(ytlen, 1)
    w = fracweights(a, tau, h, ytlen)
    fracorder = torch.matmul(w, yt)
    ed = time.time()
    ittim = ed - st
    return fracorder


def fracweights(a, tau, h, ytlen):
    w = torch.eye(ytlen)
    mem = torch.zeros(ytlen, 1)
    k = torch.arange(ytlen)
    cols = torch.arange(ytlen).unsqueeze(0).expand(ytlen, -1)
    k_st = k - torch.max(torch.tensor(0), k - torch.ceil(tau / h))
    mask = (cols >= (k - k_st + 1).unsqueeze(1)) & (cols <= (k - 1).unsqueeze(1))
    j = (k.unsqueeze(1) - cols - 1)[mask]
    w[k.long(), (k - k_st).long()] = (k_st + 2) ** (1 - a) - 3 * (k_st + 1) ** (1 - a) + 2 * k_st ** (1 - a)
    w[mask] = ((j + 2 + 1e-6) ** (1 - a) - 2 * (j + 1 + 1e-6) ** (1 - a) + (j + 1e-6) ** (1 - a))
    w[0, 0] = 0
    w = (h ** (-a) / torch.exp(torch.lgamma(2 - a))) * w
    return w


def fraccaputo_V5(yt, h, a, tau):
    device = yt.device
    ytlen = yt.shape[0]

    def fracorder_mine(a, tau, h, ytlen, yt):
        w = torch.eye(ytlen)
        k = torch.arange(ytlen)
        cols = torch.arange(ytlen).unsqueeze(0).expand(ytlen, -1)
        k_st = k - torch.max(torch.tensor(0), k - torch.ceil(tau / h))
        mask = (cols >= (k - k_st + 1).unsqueeze(1)) & (cols <= (k - 1).unsqueeze(1))
        j = torch.zeros(ytlen, ytlen)
        j = (k.unsqueeze(1) - cols - 1)[mask]
        kk = k.long()[1:]
        kk_st = k_st.long()[1:]
        w[kk.long(), (kk - kk_st).long()] = (kk_st + 1) ** (1 - a) - 3 * (kk_st) ** (1 - a) + 2 * (kk_st - 1) ** (1 - a)
        w[mask] = ((j + 2) ** (1 - a) - 2 * (j + 1) ** (1 - a) + (j) ** (1 - a))
        w[0, 0] = 0
        w = (h ** (-a) / torch.exp(torch.lgamma(2 - a))) * w
        fracorder = torch.matmul(w, yt)
        return fracorder

    # Create a differentiable wrapper for the weight calculation
    class WeightFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, tau, h, ytlen, yt):
            # w = fracweights(a, tau, h, ytlen)
            fracorder_forward = fracorder_mine(a, tau, h, ytlen, yt)
            ctx.save_for_backward(a, tau, h, torch.tensor(ytlen, device=device), yt)
            return fracorder_forward

        @staticmethod
        def backward(ctx, grad_output):
            a, tau, h, ytlen, yt = ctx.saved_tensors
            epsilon = 5 * 1e-1

            # Calculate numerical gradient for weights
            grad_fracorder_tau = torch.zeros_like(grad_output)
            fracorder_tau_1 = fracorder_mine(a, tau - epsilon, h, ytlen, yt)
            fracorder_tau_2 = fracorder_mine(a, tau + epsilon, h, ytlen, yt)
            grad_fracorder_tau = (fracorder_tau_2 - fracorder_tau_1) / (2 * epsilon)

            # grad_fracorder_alpha = torch.zeros_like(grad_output)
            # fracorder_alpha_1 = fracorder_mine(a - epsilon, tau, h, ytlen, yt)
            # fracorder_alpha_2 = fracorder_mine(a + epsilon, tau, h, ytlen, yt)
            # grad_fracorder_alpha = (fracorder_alpha_2 - fracorder_alpha_1) / (2*epsilon)

            with torch.set_grad_enabled(True):
                fracorder_a = fracorder_mine(a, tau, h, ytlen, yt)
                grad_fracorder_alpha = \
                torch.autograd.grad(fracorder_a, a, torch.ones_like(fracorder_a), create_graph=True)[0]

            # print(grad_alpha)
            # print(grad_fracorder.shape)
            # print(grad_alpha)

            # Apply chain rule to get gradient for alpha
            grad_fracorder_tau = torch.sum(grad_output * grad_fracorder_tau)
            grad_fracorder_alpha = torch.sum(grad_output * grad_fracorder_alpha)

            # print(grad_fracorder.unsqueeze(0).shape)
            # print(grad_fracorder)

            return grad_fracorder_alpha.unsqueeze(0), grad_fracorder_tau.unsqueeze(0), None, None, None

    # Calculate weights using the differentiable wrapper
    # w = WeightFunction.apply(a, tau, h, ytlen)
    fracorder = WeightFunction.apply(a, tau, h, ytlen, yt)

    # Calculate the fractional derivative
    # fracorder = torch.matmul(w, yt)

    return fracorder


def fraccaputo_fixed_tau(yt, h, a, tau_int):
    ytlen = yt.shape[0]
    w = torch.eye(ytlen, device=yt.device)
    k = torch.arange(ytlen, dtype=torch.int64, device=yt.device)

    k_st = k - torch.clamp(k - int(tau_int), min=0)

    cols = torch.arange(ytlen, dtype=torch.int64, device=yt.device).unsqueeze(0).expand(ytlen, -1)
    mask = (cols >= (k - k_st + 1).unsqueeze(1)) & (cols <= (k - 1).unsqueeze(1))
    j = (k.unsqueeze(1) - cols - 1)[mask]
    kk = k[1:]
    kk_st = k_st[1:]
    w[kk, (kk - kk_st)] = (kk_st + 1)**(1 - a) - 3 * (kk_st)**(1 - a) + 2 * (kk_st - 1)**(1 - a)
    w[mask] = (j + 2)**(1 - a) - 2 * (j + 1)**(1 - a) + j**(1 - a)
    w[0, 0] = 0
    w = (h ** (-a) / torch.exp(torch.lgamma(2 - a))) * w
    fracorder = torch.matmul(w, yt)
    return fracorder

def fraccaputo_V6_mine(yt, h, a, tau, T, tau_actual):
    tau_idx = tau / h
    tau_floor = torch.floor(tau_idx).detach()
    tau_ceil = tau_floor + 1
    if tau_floor == T/h:
        tau_ceil = tau_floor-1

    diff = tau_idx - tau_floor
    diff_1 = 1-diff

    minie = min(diff, diff_1)
    maxie = max(diff, diff_1)

    tau_floor_val = tau_floor.item()
    tau_ceil_val = tau_ceil.item()
    frac1 = fraccaputo_fixed_tau(yt, h, a, tau_floor_val)
    frac2 = fraccaputo_fixed_tau(yt, h, a, tau_ceil_val)

    # Differentiable interpolation
    fracorder = minie * frac1 + maxie * frac2
    return fracorder


def fraccaputo_new(yt, h, a, tau):
    ytlen = yt.shape[0]
    w = torch.eye(ytlen, device=yt.device)
    k = torch.arange(ytlen, dtype=torch.int64, device=yt.device)
    cols = torch.arange(ytlen, dtype=torch.int64, device=yt.device).unsqueeze(0).expand(ytlen, -1)

    # k_st = k - torch.max(torch.tensor(0),k-torch.ceil(tau/h))
    k_st = k - torch.clamp(k - smooth_ceil(tau / h), min=0)
    k_st_mine = k_st.clone().detach().long()

    mask = ((cols >= (k - k_st + 1).unsqueeze(1)) & (cols <= (k - 1).unsqueeze(1)))
    j = torch.zeros(ytlen, ytlen)
    # print(k_st.size(), mask.size(), (k.unsqueeze(1) - cols -1).size())
    j = (k.unsqueeze(1) - cols - 1)[mask]

    kk = k[1:]
    kk_st = k_st[1:]
    kk_st_mine = k_st_mine[1:]

    w[kk, (kk - kk_st_mine)] = (kk_st + 1) ** (1 - a) - 3 * (kk_st) ** (1 - a) + 2 * (kk_st - 1) ** (1 - a)
    w[mask] = ((j + 2) ** (1 - a) - 2 * (j + 1) ** (1 - a) + (j) ** (1 - a))
    w[0, 0] = 0
    w = (h ** (-a) / torch.exp(torch.lgamma(2 - a))) * w
    fracorder = torch.matmul(w, yt)
    return fracorder
