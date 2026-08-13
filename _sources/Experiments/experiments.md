# Experiments

One file per run, following the experiment template (Summary, Setup, Data, Model Design, Training, Results).

## Planned runs

- `alpha_inv.md` — recover alpha only (10k steps)
- `tau_inv.md` — recover tau only
- `joint_inv.md` — recover alpha and tau together
- `noise_sweep.md` — robustness to observation noise
- `init_sweep.md` — sensitivity to initial guess
- `batch_norm_variant.md` — batch norm on/off

## Template

- Summary: goal, one paragraph
- Date / lead
- Config: exact `config.yaml` values
- Data: generation details
- Model: network design, hyperparameters
- Training: steps, optimizer, losses
- Results: recovered alpha/tau, error, loss curves, figures
- Code: file paths (`src/fpinns/...`)