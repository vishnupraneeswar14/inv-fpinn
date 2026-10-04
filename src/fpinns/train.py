import torch
import matplotlib.pyplot as plt

from fpinns.fraccaputo import fraccaputo_V6_mine


def compute_loss(pinn, t_phy, t_obs, u_obs, freq, force_mag, m, k, c, dt, alpha, tau, T, tau_actual, lam1, lam2):
    u = pinn(t_phy)
    F_phy = force_mag*torch.sin(freq*t_phy).view(-1, 1)
    dudt = torch.autograd.grad(u, t_phy, torch.ones_like(u), create_graph=True)[0]
    dudt_frac = fraccaputo_V6_mine(u, torch.tensor([dt]), alpha, tau, T, tau_actual)
    du2dt = torch.autograd.grad(dudt, t_phy, torch.ones_like(u), create_graph=True)[0]
    loss3 = torch.mean((m*du2dt + c*dudt_frac + k*u - F_phy)**2)
    u1 = pinn(t_obs)
    loss4 = torch.mean((u1-u_obs)**2)
    loss = lam1*loss3 + lam2*loss4
    return loss


def compute_combined_loss(pinns, t_phys, t_obss, u_obss, freqs, force_mags, m, k, c, dt, alpha, tau, T, tau_actual, lam1, lam2):
    loss = None
    for pinn, t_phy, t_obs, u_obs, freq, force_mag in zip(pinns, t_phys, t_obss, u_obss, freqs, force_mags):
        l = compute_loss(pinn, t_phy, t_obs, u_obs, freq, force_mag, m, k, c, dt, alpha, tau, T, tau_actual, lam1, lam2)
        loss = l if loss is None else loss + l
    return loss


def step_alpha(pinns, optimisers, optimiser_alpha, optimiser_k, optimiser_c, t_phys, t_obss, u_obss, freqs, force_mags, m, k, c, dt, alpha, tau, T, tau_actual, lam1, lam2):
    for optimiser in optimisers:
        optimiser.zero_grad()
    optimiser_alpha.zero_grad()
    if optimiser_k is not None:
        optimiser_k.zero_grad()
    if optimiser_c is not None:
        optimiser_c.zero_grad()
    loss = compute_combined_loss(pinns, t_phys, t_obss, u_obss, freqs, force_mags, m, k, c, dt, alpha, tau, T, tau_actual, lam1, lam2)
    loss.backward()
    for optimiser in optimisers:
        optimiser.step()
    optimiser_alpha.step()
    if optimiser_k is not None:
        optimiser_k.step()
    if optimiser_c is not None:
        optimiser_c.step()
    return loss


def step_tau(pinns, optimisers, optimiser_tau, optimiser_k, optimiser_c, t_phys, t_obss, u_obss, freqs, force_mags, m, k, c, dt, alpha, tau, T, tau_actual, lam1, lam2):
    for optimiser in optimisers:
        optimiser.zero_grad()
    optimiser_tau.zero_grad()
    if optimiser_k is not None:
        optimiser_k.zero_grad()
    if optimiser_c is not None:
        optimiser_c.zero_grad()
    loss = compute_combined_loss(pinns, t_phys, t_obss, u_obss, freqs, force_mags, m, k, c, dt, alpha, tau, T, tau_actual, lam1, lam2)
    loss.backward()
    for optimiser in optimisers:
        optimiser.step()
    optimiser_tau.step()
    if optimiser_k is not None:
        optimiser_k.step()
    if optimiser_c is not None:
        optimiser_c.step()
    return loss


def step_joint(pinns, optimisers, optimiser_alpha, optimiser_tau, optimiser_k, optimiser_c, t_phys, t_obss, u_obss, freqs, force_mags, m, k, c, dt, alpha, tau, T, tau_actual, lam1, lam2):
    for optimiser in optimisers:
        optimiser.zero_grad()
    optimiser_alpha.zero_grad()
    optimiser_tau.zero_grad()
    if optimiser_k is not None:
        optimiser_k.zero_grad()
    if optimiser_c is not None:
        optimiser_c.zero_grad()
    loss = compute_combined_loss(pinns, t_phys, t_obss, u_obss, freqs, force_mags, m, k, c, dt, alpha, tau, T, tau_actual, lam1, lam2)
    loss.backward()
    for optimiser in optimisers:
        optimiser.step()
    optimiser_alpha.step()
    optimiser_tau.step()
    if optimiser_k is not None:
        optimiser_k.step()
    if optimiser_c is not None:
        optimiser_c.step()
    return loss


def init_inverse_params(alpha_init, tau_init):
    alpha = torch.tensor([alpha_init], requires_grad=True)
    tau = torch.tensor([tau_init], requires_grad=True)
    return alpha, tau


def load_checkpoint(path, pinns, alpha, tau, k=None, c=None):
    try:
        ckpt = torch.load(path, weights_only=False)
    except TypeError:
        ckpt = torch.load(path)
    for idx, pinn in enumerate(pinns):
        key = f"model_parameters{idx if idx > 0 else ''}"
        if key not in ckpt:
            key = "model_parameters"
        pinn.load_state_dict(ckpt[key])
    with torch.no_grad():
        alpha.copy_(torch.as_tensor(ckpt["inverse_parameter_alpha"], dtype=alpha.dtype))
        if "inverse_parameter_tau" in ckpt:
            tau.copy_(torch.as_tensor(ckpt["inverse_parameter_tau"], dtype=tau.dtype))
        if k is not None and "inverse_parameter_k" in ckpt:
            k.copy_(torch.as_tensor(ckpt["inverse_parameter_k"], dtype=k.dtype))
        if c is not None and "inverse_parameter_c" in ckpt:
            c.copy_(torch.as_tensor(ckpt["inverse_parameter_c"], dtype=c.dtype))


def save_checkpoint(path, pinns, alpha, tau, k=None, c=None):
    state = {"model_parameters": pinns[0].state_dict(), "inverse_parameter_alpha": alpha.detach().cpu().numpy(), "inverse_parameter_tau": tau.detach().cpu().numpy()}
    for idx in range(1, len(pinns)):
        state[f"model_parameters{idx}"] = pinns[idx].state_dict()
    if k is not None:
        state["inverse_parameter_k"] = k.detach().cpu().numpy()
    if c is not None:
        state["inverse_parameter_c"] = c.detach().cpu().numpy()
    torch.save(state, path)


def log_iteration(i, iteration_time, loss, alpha, tau, k=None, c=None):
    ktxt = f", k: {k.item()}, k_grad: {k.grad}" if k is not None else ""
    ctxt = f", c: {c.item()}, c_grad: {c.grad}" if c is not None else ""
    print(f"Training step {i}, Time taken: {iteration_time:.4f}, alpha: {alpha}, alpha_grad: {alpha.grad}, tau: {tau}, tau_grad: {tau.grad}{ktxt}{ctxt}, Total log loss : {torch.log10(torch.tensor([loss]))}")


def plot_output(ax, pinns, t_tests, t_fdms, u_fdms):
    for idx, (pinn, t_test, t_fdm, u_fdm) in enumerate(zip(pinns, t_tests, t_fdms, u_fdms)):
        u = pinn(t_test).detach()
        ax.plot(t_test[:, 0], u[:, 0], label=f"PINN{idx+1}")
        ax.plot(t_fdm, u_fdm, label=f"FDM{idx+1}", linestyle="--")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    ax.set_xlabel('Time(s)')
    ax.set_ylabel('Displacement(mm)')
    ax.set_title('Output from PINN')


def plot_alpha_param(ax, iters, alpha_list, alpha_actual):
    ax.plot(iters, alpha_list)
    ax.hlines(alpha_actual, 0, len(alpha_list), colors=['g'])
    ax.legend(["PINN", "Exact"])
    ax.set_xlabel('Iters')
    ax.set_ylabel('alpha')
    ax.set_title('Prediction of Fractional Order alpha')


def plot_tau_param(ax, iters, tau_list, tau_actual):
    ax.plot(iters, tau_list)
    ax.hlines(tau_actual, 0, len(tau_list), colors=['g'])
    ax.legend(["PINN", "Exact"])
    ax.set_xlabel('Iters')
    ax.set_ylabel('tau')
    ax.set_title('Prediction of Fractional Order tau')


def plot_kc_param(ax, iters, k_list, c_list, k_actual, c_actual):
    ax.plot(iters, k_list, label="k")
    ax.plot(iters, c_list, label="c")
    ax.hlines(k_actual, 0, len(k_list), colors=['g'], linestyles='dotted')
    ax.hlines(c_actual, 0, len(c_list), colors=['r'], linestyles='dotted')
    ax.legend()
    ax.set_xlabel('Iters')
    ax.set_ylabel('k, c')
    ax.set_title('Prediction of Stiffness k and Damping c')


def plot_loss(ax, iters, l):
    ax.plot(iters, torch.log10(torch.tensor([l])).view(-1))
    ax.set_xlabel('Iters')
    ax.set_ylabel('Log Loss')
    ax.set_title('Loss Curve')


def plot_alpha(i, iters, alpha_list, alpha_actual, k_list, c_list, k_actual, c_actual, pinns, t_tests, t_fdms, u_fdms, l, save_path, fig_size):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=fig_size)
    plot_output(ax1, pinns, t_tests, t_fdms, u_fdms)
    plot_alpha_param(ax2, iters, alpha_list, alpha_actual)
    plot_kc_param(ax3, iters, k_list, c_list, k_actual, c_actual)
    plot_loss(ax4, iters, l)
    fig.suptitle(f"Training step {i}")
    plt.savefig(save_path)
    return fig

#change for git to reflect


def plot_tau(i, iters, tau_list, tau_actual, k_list, c_list, k_actual, c_actual, pinns, t_tests, t_fdms, u_fdms, l, save_path, fig_size):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=fig_size)
    plot_output(ax1, pinns, t_tests, t_fdms, u_fdms)
    plot_tau_param(ax2, iters, tau_list, tau_actual)
    plot_kc_param(ax3, iters, k_list, c_list, k_actual, c_actual)
    plot_loss(ax4, iters, l)
    fig.suptitle(f"Training step {i}")
    plt.savefig(save_path)
    return fig


def plot_joint(i, iters, alpha_list, alpha_actual, tau_list, tau_actual, k_list, c_list, k_actual, c_actual, pinns, t_tests, t_fdms, u_fdms, l, save_path, fig_size):
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=fig_size)
    plot_output(ax1, pinns, t_tests, t_fdms, u_fdms)
    plot_alpha_param(ax2, iters, alpha_list, alpha_actual)
    plot_tau_param(ax3, iters, tau_list, tau_actual)
    plot_kc_param(ax4, iters, k_list, c_list, k_actual, c_actual)
    plot_loss(ax5, iters, l)
    fig.suptitle(f"Training step {i}")
    plt.savefig(save_path)
    return fig


def plot_kc(i, iters, k_list, c_list, k_actual, c_actual, save_path, fig_size):
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    plot_kc_param(ax, iters, k_list, c_list, k_actual, c_actual)
    fig.suptitle(f"Training step {i}")
    plt.savefig(save_path)
    return fig