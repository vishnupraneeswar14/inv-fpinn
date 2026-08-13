import os
import time

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

from cli import parse_args
from fdm import FracSDOF
from ffn import Net
from train import (init_inverse_params, load_checkpoint, log_iteration, plot_alpha,
                   plot_tau, save_checkpoint, step_alpha, step_tau)

cfg = parse_args()

model_cfg = cfg["model"]
phys_cfg = cfg["physics"]
train_cfg = cfg["training"]
art_cfg = cfg["artifacts"]

torch.manual_seed(train_cfg["seed"])
freq = model_cfg["freq"]
pinn = Net(model_cfg["input_size"], model_cfg["hidden_size"], model_cfg["output_size"],
           model_cfg["layer_count"], freq, model_cfg["activation"], model_cfg["activation_init"],
           model_cfg["use_mask"])
m, k, c = phys_cfg["m"], phys_cfg["k"], phys_cfg["c"]
xo, vo = phys_cfg["x0"], phys_cfg["v0"]

alpha_actual = phys_cfg["alpha_actual"]
tau_actual = phys_cfg["tau_actual"]
T = phys_cfg["T"]

alpha, tau = init_inverse_params(train_cfg["alpha_init"], train_cfg["tau_init"])


resume_path = train_cfg["resume_path"]
if resume_path:
    load_checkpoint(resume_path, pinn, alpha, tau)

dt = phys_cfg["dt"]
t_phy = torch.linspace(0, T, train_cfg["t_phy_points"], requires_grad=True).view(-1, 1)
t_test = torch.linspace(0, T, train_cfg["t_test_points"]).view(-1, 1)
t = np.arange(0, T, dt)

F = phys_cfg["force_mag"]*np.sin(freq*t)
u_fdm, t_fdm = FracSDOF(m, k, c, dt, F, xo, vo, T, alpha_actual, tau_actual)
num_indices = train_cfg["t_obs_points"]
index = np.linspace(0, t.shape[0] - 1, num_indices, dtype=int)
t_obs = torch.tensor(t[index], dtype=torch.float32).view(-1, 1)
u_obs = torch.tensor(u_fdm[index]).view(-1, 1) + train_cfg["noise_std"] * torch.randn_like(t_obs)
images = []

optimiser       = torch.optim.Adam(list(pinn.parameters()), lr=train_cfg["lr_pinn"])
optimiser_tau   = torch.optim.Adam([tau], lr=train_cfg["lr_tau"])
optimiser_alpha = torch.optim.Adam([alpha], lr=train_cfg["lr_alpha"])

mode = train_cfg["mode"]
if mode == "alpha":
    phases = [("alpha", train_cfg["alpha_steps"])]
elif mode == "tau":
    phases = [("tau", train_cfg["tau_steps"])]
elif mode == "alpha_tau":
    phases = [("alpha", train_cfg["alpha_steps"]), ("tau", train_cfg["tau_steps"])]
elif mode == "tau_alpha":
    phases = [("tau", train_cfg["tau_steps"]), ("alpha", train_cfg["alpha_steps"])]
else:
    raise ValueError(f"Unknown mode: {mode}")

plot_every = art_cfg["plot_every"]
gif_every = art_cfg["gif_every"]
ckpt_every = art_cfg["ckpt_every"]
fig_size = tuple(art_cfg["fig_size"])

save_dir = art_cfg["save_dir"]
os.makedirs(save_dir, exist_ok=True)
jpg_alpha_path = os.path.join(save_dir, art_cfg["jpg_alpha_name"].format(alpha_actual=alpha_actual, mode=mode))
jpg_tau_path = os.path.join(save_dir, art_cfg["jpg_tau_name"].format(alpha_actual=alpha_actual, tau_actual=tau_actual, mode=mode))
gif_path = os.path.join(save_dir, art_cfg["gif_name"].format(alpha_actual=alpha_actual, mode=mode))
pth_path = os.path.join(save_dir, art_cfg["pth_name"].format(alpha_actual=alpha_actual, mode=mode))
pth_alpha_path = os.path.join(save_dir, art_cfg["pth_alpha_name"].format(alpha_actual=alpha_actual, mode=mode))
pth_tau_path = os.path.join(save_dir, art_cfg["pth_tau_name"].format(alpha_actual=alpha_actual, mode=mode))

lam1, lam2 = train_cfg["lam1"], train_cfg["lam2"]

for phase_idx, (phase_name, steps) in enumerate(phases):
    iters = []
    l = []
    alpha_list = []
    tau_list = []
    for i in range(steps):
        iters.append(i)
        st = time.time()

        if phase_name == "alpha":
            loss = step_alpha(pinn, optimiser, optimiser_alpha, t_phy, t_obs, u_obs, freq, phys_cfg["force_mag"], m, k, c, dt,
                              alpha, tau, T, tau_actual, lam1, lam2)
            with torch.no_grad():
                alpha.data = torch.clamp(alpha.data, train_cfg["clamp_min"], train_cfg["clamp_max"])
                alpha_list.append(alpha.item())
        else:
            loss = step_tau(pinn, optimiser, optimiser_tau, t_phy, t_obs, u_obs, freq, phys_cfg["force_mag"], m, k, c, dt,
                            alpha, tau, T, tau_actual, lam1, lam2)
            with torch.no_grad():
                tau_list.append(tau.item())
        l.append(loss.detach())

        iteration_time = time.time() - st

        if i % plot_every == 0:
            log_iteration(i, iteration_time, loss, alpha, tau)
            if phase_name == "alpha":
                fig = plot_alpha(i, iters, alpha_list, alpha_actual, pinn, t_test, t_fdm, u_fdm, l,
                                 jpg_alpha_path, fig_size)
            else:
                fig = plot_tau(i, iters, tau_list, tau_actual, pinn, t_test, t_fdm, u_fdm, l,
                               jpg_tau_path, fig_size)
            fig.canvas.draw()
            image_rgba = fig.canvas.buffer_rgba()
            width, height = fig.canvas.get_width_height()
            image_rgb = np.frombuffer(image_rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
            images.append(image_rgb)
            plt.close(fig)

        if i % gif_every == 0:
            imageio.mimsave(gif_path, images, fps=art_cfg["fps"])
        if i % ckpt_every == 0:
            save_checkpoint(pth_path, pinn, alpha, tau)

    if len(phases) == 2 and phase_idx == 0:
        if phase_name == "alpha":
            save_checkpoint(pth_alpha_path, pinn, alpha, tau)
        else:
            save_checkpoint(pth_tau_path, pinn, alpha, tau)

save_checkpoint(pth_path, pinn, alpha, tau)