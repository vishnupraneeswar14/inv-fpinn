# Problem Formulation

## Sections to fill

- Fractional SDOF oscillator: `m * d2u/dt2 + k * u + c * D^alpha u = F(t)`
- Caputo fractional derivative definition (memory kernel, tau)
- Forward problem vs inverse problem
- Inverse problem: recover alpha, tau from noisy displacement observations
- PINN setup: physics residual loss + data fit loss, two-stage training (alpha steps, tau steps)
- Reference solution via FDM (`src/fpinns/fdm.py`) as sanity check