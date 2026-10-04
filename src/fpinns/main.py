import os
import time

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

from fpinns.cli import parse_args
from fpinns.fdm import ExpSDOF, FracSDOF, fdm_no_fractional
from fpinns.ffn import Net
from fpinns.train import init_inverse_params, load_checkpoint, log_iteration, plot_alpha, plot_joint, plot_kc, plot_tau, save_checkpoint, step_alpha, step_joint, step_tau

cfg = parse_args()

model_cfg = cfg["model"]
phys_cfg = cfg["physics"]
train_cfg = cfg["training"]
art_cfg = cfg["artifacts"]

torch.manual_seed(train_cfg["seed"])
m = phys_cfg["m"]
k0, c0 = float(phys_cfg["k_actual"]), float(phys_cfg["c_actual"])

alpha_actual = phys_cfg["alpha_actual"]
tau_actual = phys_cfg["tau_actual"]
T = phys_cfg["T"]
dt = phys_cfg["dt"]

alpha, tau = init_inverse_params(train_cfg["alpha_init"], train_cfg["tau_init"])
train_KC = train_cfg.get("k_c_trainable", False)
k = torch.tensor([float(train_cfg["k_initial"])], requires_grad=train_KC)
c = torch.tensor([float(train_cfg["c_initial"])], requires_grad=train_KC)

expt_cfgs = cfg["fdm_signals"]
pinns = []
for e in expt_cfgs:
    pinns.append(Net(model_cfg["input_size"], model_cfg["hidden_size"], model_cfg["output_size"], model_cfg["layer_count"], e["freq"], model_cfg["activation"], model_cfg["activation_init"], model_cfg["use_mask"]))

resume_path = train_cfg["resume_path"]
if resume_path:
    load_checkpoint(resume_path, pinns, alpha, tau, k, c)

print(f'alpha_actual: {alpha_actual}, tau_actual: {tau_actual}, T: {T}, alpha: {alpha}, tau: {tau}, k: {k.item()}, c: {c.item()}')

t = np.arange(0, T, dt)
num_indices = train_cfg["t_obs_points"]
index = np.linspace(0, t.shape[0] - 1, num_indices, dtype=int)

t_phys, t_tests, t_obss, u_obss, u_fdms, t_fdms, freqs, force_mags = [], [], [], [], [], [], [], []
for e in expt_cfgs:
    F = e["force_mag"]*np.sin(e["freq"]*t)
    x0_e, v0_e = e.get("x0", 0.0), e.get("v0", 0.0)
    if phys_cfg.get("fdm_exp", False):
        u_fdm, t_fdm = ExpSDOF(m, k0, c0, dt, F, x0_e, v0_e, T, phys_cfg["tauc"], tau_actual)
    elif phys_cfg.get("fdm_no_fractional", False):
        u_fdm, t_fdm = fdm_no_fractional(m, k0, c0, dt, F, x0_e, v0_e, T)
    else:
        u_fdm, t_fdm = FracSDOF(m, k0, c0, dt, F, x0_e, v0_e, T, alpha_actual, tau_actual)
    t_phys.append(torch.linspace(0, T, train_cfg["t_phy_points"], requires_grad=True).view(-1, 1))
    t_tests.append(torch.linspace(0, T, train_cfg["t_test_points"]).view(-1, 1))
    t_obs = torch.tensor(t[index], dtype=torch.float32).view(-1, 1)
    u_obs = torch.tensor(u_fdm[index]).view(-1, 1) + train_cfg["noise_std"] * torch.randn_like(t_obs)
    t_obss.append(t_obs)
    u_obss.append(u_obs)
    u_fdms.append(u_fdm)
    t_fdms.append(t_fdm)
    freqs.append(e["freq"])
    force_mags.append(e["force_mag"])
images = []
kc_images = []

optimisers = [torch.optim.Adam(list(p.parameters()), lr=train_cfg["lr_pinn"]) for p in pinns]
optimiser_tau   = torch.optim.Adam([tau], lr=train_cfg["lr_tau"])
optimiser_alpha = torch.optim.Adam([alpha], lr=train_cfg["lr_alpha"])
optimiser_k = torch.optim.Adam([k], lr=train_cfg["lr_k"]) if train_KC else None
optimiser_c = torch.optim.Adam([c], lr=train_cfg["lr_c"]) if train_KC else None

mode = train_cfg["mode"]
if mode == "alpha":
    phases = [("alpha", train_cfg["alpha_steps"])]
elif mode == "tau":
    phases = [("tau", train_cfg["tau_steps"])]
elif mode == "alpha_tau":
    phases = [("alpha", train_cfg["alpha_steps"]), ("tau", train_cfg["tau_steps"])]
elif mode == "tau_alpha":
    phases = [("tau", train_cfg["tau_steps"]), ("alpha", train_cfg["alpha_steps"])]
elif mode == "joint":
    phases = [("joint", train_cfg["alpha_tau_steps"])]
else:
    raise ValueError(f"Unknown mode: {mode}")

plot_every = art_cfg["plot_every"]
gif_every = art_cfg["gif_every"]
ckpt_every = art_cfg["ckpt_every"]
fig_size = tuple(art_cfg["fig_size"])

save_dir = art_cfg["save_dir"]
os.makedirs(save_dir, exist_ok=True)

lam1, lam2 = train_cfg["lam1"], train_cfg["lam2"]

stem = "mask_step" if model_cfg["use_mask"] else "step"


def checkpoint_path(step, phase, stem, save_dir, ckpt_dir):
    return os.path.join(save_dir, ckpt_dir, phase, f"{stem}_{step}.pth")


def plot_path(step, phase, stem, save_dir, plots_dir):
    return os.path.join(save_dir, plots_dir, phase, f"{stem}.jpg")

for phase_idx, (phase_name, steps) in enumerate(phases):
    iters = []
    l = []
    alpha_list = []
    tau_list = []
    k_list = []
    c_list = []
    for dir_key in ("ckpt_dir", "plots_dir", "gif_dir"):
        os.makedirs(os.path.join(save_dir, art_cfg[dir_key], phase_name), exist_ok=True)
    print(f'Phase {phase_name} will be run for {steps} steps...')

    for i in range(steps):
        iters.append(i)
        st = time.time()

        if phase_name == "joint":
            loss = step_joint(pinns, optimisers, optimiser_alpha, optimiser_tau, optimiser_k, optimiser_c, t_phys, t_obss, u_obss, freqs, force_mags, m, k, c, dt, alpha, tau, T, tau_actual, lam1, lam2)
            with torch.no_grad():
                alpha.data = torch.clamp(alpha.data, train_cfg["clamp_min"], train_cfg["clamp_max"])
                alpha_list.append(alpha.item())
                tau_list.append(tau.item())
        elif phase_name == "alpha":
            loss = step_alpha(pinns, optimisers, optimiser_alpha, optimiser_k, optimiser_c, t_phys, t_obss, u_obss, freqs, force_mags, m, k, c, dt, alpha, tau, T, tau_actual, lam1, lam2)
            with torch.no_grad():
                alpha.data = torch.clamp(alpha.data, train_cfg["clamp_min"], train_cfg["clamp_max"])
                alpha_list.append(alpha.item())
        else:
            loss = step_tau(pinns, optimisers, optimiser_tau, optimiser_k, optimiser_c, t_phys, t_obss, u_obss, freqs, force_mags, m, k, c, dt, alpha, tau, T, tau_actual, lam1, lam2)
            with torch.no_grad():
                tau_list.append(tau.item())
        with torch.no_grad():
            k_list.append(k.item())
            c_list.append(c.item())
        l.append(loss.detach())

        iteration_time = time.time() - st

        if i % plot_every == 0:
            log_iteration(i, iteration_time, loss, alpha, tau, k, c)
            jpg_path = plot_path(i, phase_name, stem, save_dir, art_cfg["plots_dir"])
            os.makedirs(os.path.dirname(jpg_path), exist_ok=True)
            if phase_name == "joint":
                fig = plot_joint(i, iters, alpha_list, alpha_actual, tau_list, tau_actual, k_list, c_list, k0, c0, pinns, t_tests, t_fdms, u_fdms, l, jpg_path, fig_size)
            elif phase_name == "alpha":
                fig = plot_alpha(i, iters, alpha_list, alpha_actual, k_list, c_list, k0, c0, pinns, t_tests, t_fdms, u_fdms, l, jpg_path, fig_size)
            else:
                fig = plot_tau(i, iters, tau_list, tau_actual, k_list, c_list, k0, c0, pinns, t_tests, t_fdms, u_fdms, l, jpg_path, fig_size)

            fig.canvas.draw()
            image_rgba = fig.canvas.buffer_rgba()
            width, height = fig.canvas.get_width_height()
            image_rgb = np.frombuffer(image_rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
            images.append(image_rgb)
            plt.close(fig)

            kc_jpg = os.path.join(os.path.dirname(jpg_path), "kc.jpg")
            kc_fig = plot_kc(i, iters, k_list, c_list, k0, c0, kc_jpg, fig_size)
            kc_fig.canvas.draw()
            kc_image_rgba = kc_fig.canvas.buffer_rgba()
            kc_width, kc_height = kc_fig.canvas.get_width_height()
            kc_image_rgb = np.frombuffer(kc_image_rgba, dtype=np.uint8).reshape(kc_height, kc_width, 4)[:, :, :3]
            kc_images.append(kc_image_rgb)
            plt.close(kc_fig)

        if i % gif_every == 0:
            gif_path = os.path.join(save_dir, art_cfg["gif_dir"], phase_name, f"{stem}_{phase_name}.gif")
            os.makedirs(os.path.dirname(gif_path), exist_ok=True)
            imageio.mimsave(gif_path, images, fps=art_cfg["fps"])
            kc_gif_path = os.path.join(save_dir, art_cfg["gif_dir"], phase_name, f"{stem}_{phase_name}_kc.gif")
            imageio.mimsave(kc_gif_path, kc_images, fps=art_cfg["fps"])
        if i % ckpt_every == 0:
            save_checkpoint(checkpoint_path(i, phase_name, stem, save_dir, art_cfg["ckpt_dir"]), pinns, alpha, tau, k, c)

    print(f'Phase {phase_name} is done with {steps} steps!')
    if len(phases) == 2 and phase_idx == 0:
        save_checkpoint(checkpoint_path(steps - 1, phase_name, stem, save_dir, art_cfg["ckpt_dir"]), pinns, alpha, tau, k, c)

save_checkpoint(checkpoint_path(steps - 1, phase_name, stem, save_dir, art_cfg["ckpt_dir"]), pinns, alpha, tau, k, c)

#change for git to reflect
