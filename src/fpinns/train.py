import torch
import matplotlib.pyplot as plt

from fraccaputo import fraccaputo_V6_mine


def compute_loss(pinn, t_phy, t_obs, u_obs, freq, force_mag, m, k, c, dt, alpha, tau, T, tau_actual, lam1, lam2):
    u = pinn(t_phy)
    F_phy = force_mag*torch.sin(freq*t_phy).view(-1, 1)
    dudt = torch.autograd.grad(u, t_phy, torch.ones_like(u), create_graph=True)[0]
    dudt_frac = fraccaputo_V6_mine(u, torch.tensor([dt]), alpha, tau, T, tau_actual)
    du2dt = torch.autograd.grad(dudt, t_phy, torch.ones_like(u), create_graph=True)[0]
    loss3 = torch.mean((m*du2dt + k*u + c*dudt_frac - F_phy)**2)
    u1 = pinn(t_obs)
    loss4 = torch.mean((u1-u_obs)**2)
    loss = lam1*loss3 + lam2*loss4
    return loss


def step_alpha(pinn, optimiser, optimiser_alpha, t_phy, t_obs, u_obs, freq, force_mag, m, k, c, dt, alpha, tau, T, tau_actual, lam1, lam2):
    optimiser.zero_grad()
    optimiser_alpha.zero_grad()
    loss = compute_loss(pinn, t_phy, t_obs, u_obs, freq, force_mag, m, k, c, dt, alpha, tau, T, tau_actual, lam1, lam2)
    loss.backward()
    optimiser.step()
    optimiser_alpha.step()
    return loss


def step_tau(pinn, optimiser, optimiser_tau, t_phy, t_obs, u_obs, freq, force_mag, m, k, c, dt, alpha, tau, T, tau_actual, lam1, lam2):
    optimiser.zero_grad()
    optimiser_tau.zero_grad()
    loss = compute_loss(pinn, t_phy, t_obs, u_obs, freq, force_mag, m, k, c, dt, alpha, tau, T, tau_actual, lam1, lam2)
    loss.backward()
    optimiser.step()
    optimiser_tau.step()
    return loss


def init_inverse_params(alpha_init, tau_init):
    alpha = torch.tensor([alpha_init], requires_grad=True)
    tau = torch.tensor([tau_init], requires_grad=True)
    return alpha, tau


def load_checkpoint(path, pinn, alpha, tau):
    ckpt = torch.load(path, weights_only=False)
    pinn.load_state_dict(ckpt["model_parameters"])
    with torch.no_grad():
        alpha.copy_(torch.as_tensor(ckpt["inverse_parameter_alpha"], dtype=alpha.dtype))
        if "inverse_parameter_tau" in ckpt:
            tau.copy_(torch.as_tensor(ckpt["inverse_parameter_tau"], dtype=tau.dtype))


def save_checkpoint(path, pinn, alpha, tau):
    torch.save({"model_parameters": pinn.state_dict(),
                "inverse_parameter_alpha": alpha.detach().cpu().numpy(),
                "inverse_parameter_tau": tau.detach().cpu().numpy()}, path)


def log_iteration(i, iteration_time, loss, alpha, tau):
    print(f"Training step {i}, Time taken: {iteration_time:.4f}, alpha: {alpha}, alpha_grad: {alpha.grad}, tau: {tau}, tau_grad: {tau.grad}, Total log loss : {torch.log10(torch.tensor([loss]))}")


def plot_output(ax, pinn, t_test, t_fdm, u_fdm):
    u = pinn(t_test).detach()
    ax.plot(t_test[:, 0], u[:, 0])
    ax.plot(t_fdm, u_fdm)
    ax.legend(["PINN", "FDM"])
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


def plot_loss(ax, iters, l):
    ax.plot(iters, torch.log10(torch.tensor([l])).view(-1))
    ax.set_xlabel('Iters')
    ax.set_ylabel('Log Loss')
    ax.set_title('Loss Curve')


def plot_alpha(i, iters, alpha_list, alpha_actual, pinn, t_test, t_fdm, u_fdm, l, save_path, fig_size):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=fig_size)
    plot_output(ax1, pinn, t_test, t_fdm, u_fdm)
    plot_alpha_param(ax2, iters, alpha_list, alpha_actual)
    plot_loss(ax3, iters, l)
    fig.suptitle(f"Training step {i}")
    plt.savefig(save_path)
    return fig


def plot_tau(i, iters, tau_list, tau_actual, pinn, t_test, t_fdm, u_fdm, l, save_path, fig_size):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=fig_size)
    plot_output(ax1, pinn, t_test, t_fdm, u_fdm)
    plot_tau_param(ax2, iters, tau_list, tau_actual)
    plot_loss(ax3, iters, l)
    fig.suptitle(f"Training step {i}")
    plt.savefig(save_path)
    return fig