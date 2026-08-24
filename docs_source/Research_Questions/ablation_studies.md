# Ablation Studies

## 1. Summary

Controlled experiments that decide what goes into the final model: train a small model, change exactly one thing, compare.

```
baseline run -> change ONE thing -> compare -> winner becomes new baseline -> repeat for next param
```

## 2. Why

- Attribution: a full run may prove working of the whole model. An ablation proves which part carries the value.
- Cheap kills: hurts at small scale -> dead at large scale. Works at small scale -> confirm if same trend holds true for larger model.

## 3. Scale

Two ways to scale down:

1. **Same model, fewer tokens** — target model trained on 100B tokens instead of the full final training tokens 11T.
2. **Smaller proxy model** — when the target is too big, train a smaller stand-in (eg: 1B ablation model for 3B full model).


## 4. Rules

- Change one thing per run. Two changes -> you can't explicitly tell which caused the effect.
- Same seeds, same data, same steps.
- Track parameter counts to keep the comparison fair (Ideally maintain almost same parameters b/w diff ablation configs).
- Small gaps can be noise. Re-run winners with different seeds to see how much results vary.
- Skip what doesn't matter: if a change doesn't help your use case or your training, don't test it.

## 5. Evaluation

Loss is not the metric:

```
Wikipedia text -> lower loss -> worse model
```

Losses also aren't comparable across tokenizer changes.

Pick benchmarks with four properties:

- **Monotonic:** score improves as training goes on
- **Low noise:** same setup, different seeds, similar scores
- **Above random:** useless if the model scores randomly for thousands of steps
- **Ranking consistent:** the early leader stays the leader

Small ablations use cloze-formulation tasks. Multiple-choice needs too many tokens to leave random territory.

## 6. Inverse modelling with fPINNs

Problem: recover alpha and tau of a fractional SDOF oscillator from noisy displacement observations.

```
m * d2u/dt2 + k * u + c * D^alpha u = F(t)      # physics residual loss (Caputo derivative)
recovered (alpha, tau) vs actual values         # data fit loss
```

Potential ablations can be

- **Forward first, inverse later.** Fit the forward problem (known alpha, tau) before touching the inverse. A change that hurts the forward fit -> dead for inverse too. Forward = small scale, inverse = final run.
- **Two-stage training (alpha steps, tau steps):** ablate each stage separately. Change the alpha update scheme -> keep the tau stage fixed.
- **Noise:** observations carry measurement noise. Re-run winners with different seeds and noise realisations to check the gap isn't noise.
- **FDM reference** (`src/fpinns/fdm.py`) is the eval-suite validation: the PINN must reproduce the FDM solution before you trust it on the inverse problem.
- **Metric = recovered vs actual error.** Monotonic: error drops as training goes on. Ranking consistent: the scheme that recovers alpha best early stays best. If not, the ablation setup is broken, not the model.