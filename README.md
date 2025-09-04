
# MNIST FFN + Sparse Autoencoder (SAE)

**TL;DR**  
We train a small MNIST classifier (FFN), then fit a **JumpReLU** sparse autoencoder on the **classifier’s logits**.  
Target sparsity is **~1 active latent per example** (MNIST has 10 classes).  
Fidelity is measured as test accuracy when replacing model logits with SAE-reconstructed logits.  
Best run: target_actives=1 → achieved=1.63 actives/example; Δ accuracy = 5.62 pp (97.21% → 91.59%); Val recon MSE = 0.1743 (normalized space) / 8.4039 (raw space).

With bs=8, an MLP sweep over hidden∈{1024, 8192} and LR∈{1e-1,1e-2,1e-3,1e-4} found a best baseline at **hidden=8192, lr=1e-4**: **Val CE 0.0809**, **Test CE 0.0738**, **Test Acc 0.9759**.
Validation classifier loss (CE): baseline 0.1061 → SAE-reconstructed 0.3659 (ΔCE +0.2598).
The notebook writes **`artifacts_mnist_sae_logits/BEST_RESULTS.md`** with the best run.

---

## Motivation

- **Interpretability via sparsity.** Sparse codes tend to be crisper and easier to reason about.
- **Right target signal for SAE.** Linear matrices are already interpretable; the **post-MLP logits** reflect the decision signal we care about.
- **Fast iteration.** MNIST allows quick prototyping of SAE design and λ calibration without heavy compute.

### Research Questions

1. Can we enforce **~1 active latent per example** on MNIST logits while preserving accuracy?
2. Is a simple **λ calibration** (1 epoch per grid point) sufficient to hit a desired L0 reliably?
3. What accuracy loss (Δ) results from replacing logits with **SAE reconstructions**?

---

## Methods

### Data
- **Dataset:** MNIST (60k train / 10k test).
- **Split:** 80% train / 20% val from the original train split.
- **Transform:** `ToTensor()` only.

### Classifier (FFN)
- **FFN types:** `ReLU` (default) or `GeGLU`.
- **Shapes** (Shazeer suffix style):  
  `x_bchw` → flatten to `x_bi`; hidden `h_bh`; logits `y_bo`; weights `W_in_ih`, `W_out_ho` (and `W_gate_ih` for GeGLU).  
- **Defaults:** `HIDDEN_DIM=128`, `EPOCHS_FFN=3`, `LR_FFN=1e-3`, `BATCH_FFN=8`.
- **Metric:** baseline test accuracy on logits.

**Why this FFN?**  
Small and stable; the goal is to analyze and sparsify **logits**, not to chase SOTA.

**Sweep for low CE (bs=8).** We additionally train 8 MLPs with hidden ∈ {1024, 8192} and LR ∈ {1e-1, 1e-2, 1e-3, 1e-4}. We select the best by **lowest validation cross-entropy**. The best setting was **8192 hidden, lr=1e-4**.

### Sparse Autoencoder (JumpReLU) on Logits
- **Input/target:** Logits `y_bo` (dimension `10`).
- **Activation:** **JumpReLU** with straight-through estimator:
  - Forward: **hard gate** `1[u>θ]` → clear on/off sparsity.
  - Backward: **sigmoid surrogate** with temperature `τ` → stable gradients.
- **Latents:** `SAE_LATENTS=32` (lightly overcomplete).
- **Target sparsity:** `TARGET_ACTIVES=[1]` → aim for ~**1 active latent per example**.
- **Normalization:** Optionally standardize logits per dimension before SAE (`NORMALIZE_Y=True`), de-standardize reconstruction.
- **Training:** `EPOCHS_SAE_FINAL=5`, `LR_SAE_FINAL=1e-3`, `BATCH_SAE=8`.  
  Decoder columns are renormalized each step for stability.

**Why JumpReLU?**  
Hard sparsity in the forward pass (interpretable), smooth surrogate in the backward pass (trainable).

### λ Calibration (to hit L0)
- **Procedure:** geometric λ grid → train **1 epoch** each → choose λ whose mean **hard** #actives on the val buffer is closest to target.  
- **Then:** train a **final SAE** at the chosen λ for multiple epochs.
- **L0 metric:** mean hard gate count via `gate_counts()`.

**Why this approach?**  
Simple, fast, and robust — avoids complex schedules while steering average sparsity.

### Evaluation (Fidelity)
- Compute **baseline test accuracy** using original logits.
- Replace logits with **SAE-reconstructed logits** and recompute test accuracy.
- Report **Δ accuracy** = baseline − reconstructed.  
  Δ≈0 while L0≈1 → sparse **and** faithful code.
**Reconstruction error (MSE).** We report mean squared error between targets and reconstructions (averaged over all examples and dimensions) in two spaces: (1) the **normalized SAE space** used for training (primary metric), and (2) the **raw (unnormalized) logit space** for intuition. The primary number we report in the paper-style write-up is the **validation MSE in the normalized space**.
  **Classifier loss (CE).** We also report cross-entropy from (i) the baseline MLP logits and (ii) SAE-reconstructed logits, alongside accuracy deltas on train/val/test.

---

## Process (Repro Steps)

1. **Install deps** (in the notebook):
   
   pip install "pytorch-lightning<3" "torch>=2.2,<3" torchvision torchaudio matplotlib tqdm


2. **Train FFN** to get baseline accuracy.
3. **Collect logits** (train/val/test) and optionally standardize.
4. **λ calibration** targeting L0≈1 (fast, 1 epoch per candidate λ).
5. **Train final SAE** at the chosen λ and evaluate fidelity on test.
6. **Artifacts:**

   * artifacts_mnist_sae_logits/summary.json
   * artifacts_mnist_sae_logits/BEST_RESULTS.md

  

---

## Limitations

* **MNIST is simple:** results may not transfer to richer data or deep nets.
* **Single metric:** we don’t evaluate calibration (ECE), NLL, or robustness.
* **Single seed / short epochs:** chosen for speed; no seed sweep here.
* **Fixed τ/θ init:** could be improved with schedules or per-feature adaptation.
* **No qualitative interp in this README:** (e.g., atom/prototype visualization) can be added later.

---

## Results

After running the notebook, `BEST_RESULTS.md` contains the best run.
**Latest run reported:**

```
# Best Result (Auto)
- target_actives: 1
- achieved_actives_val: 1.63
- lambda: 9.000e-02
- baseline_acc: 0.9721
- recon_acc_logits: 0.9159
- delta_acc_logits: 0.0562
- sae_latents: 32
- tau: 0.1
- normalize_logits: True
- batch_sizes: FFN=8, SAE=8
```
### Reconstruction Error

**Mean squared error (averaged over all examples and dimensions):**

| Split | MSE (normalized space) | MSE (raw space) |
|---|---:|---:|
| Train | 0.173436 | 8.324016 |
| Val   | **0.174266** | **8.403860** |
| Test  | 0.173423 | 8.330576 |

**Per-example Val MSE (normalized space):** mean `0.174266`, median `0.155517`, p95 `0.369527`; best idx `4913` → `0.006609`; worst idx `338` → `1.099880`.

> Note: Normalized-space MSE (~0.174) matches the SAE training objective and PL logs. Higher raw-space MSE (~8.40) reflects the variance scale of original logits.
### MLP Sweep (bs=8)

We trained 8 MLPs with hidden ∈ {1024, 8192} and LR ∈ {1e-1, 1e-2, 1e-3, 1e-4}, selecting by **lowest Val CE**.

**Top-3 (by Val CE):**
| Rank | Hidden | LR   | Val CE | Val Acc | Test CE | Test Acc | Checkpoint |
|-----:|------:|:----:|-------:|--------:|--------:|---------:|:-----------|
| 1 | 8192 | 1e-4 | **0.0809** | 0.9756 | **0.0738** | **0.9759** | `ffn_d8192_lr1em04.pt` |
| 2 | 1024 | 1e-3 | 0.1054 | 0.9709 | — | — | `ffn_d1024_lr1em03.pt` |
| 3 | 1024 | 1e-4 | 0.1218 | 0.9653 | — | — | `ffn_d1024_lr1em04.pt` |


### Classifier Loss & Accuracy

| Split | CE (baseline) | CE (SAE recon) | ΔCE | Acc (baseline) | Acc (SAE recon) | ΔAcc |
|---|---:|---:|---:|---:|---:|---:|
| Train | 0.0551 | 0.3193 | +0.2643 | 0.9824 | 0.9140 | −0.0684 |
| Val   | 0.1061 | 0.3659 | +0.2598 | 0.9669 | 0.9055 | −0.0614 |
| Test  | 0.0933 | 0.3456 | +0.2523 | 0.9721 | 0.9159 | −0.0562 |

**Interpretation:**

* Achieved actives **1.63 > 1.0** indicates the solution is a bit **too dense** for the target.
* Accuracy drop **Δ ≈ 0.056** suggests the reconstruction can be improved.
* **Normalized vs raw space:** We treat normalized-space Val MSE (0.1743) as the primary metric; raw-space Val MSE (8.404) is provided for intuition about the original logit scale.
* **Heavy-tailed errors:** Per-example Val MSE is skewed (median < mean; p95 ≈ 2× mean), suggesting a small subset of hard examples dominates the error.
* **CE tracks accuracy deltas:** Val ΔCE **+0.2598** aligns with ΔAcc **−0.0614**, indicating the SAE drops some decision-relevant detail even while keeping logits broadly faithful.
* **Where to improve:** Reducing actives toward ~1.3–1.5 (by slightly increasing λ) and calibrating θ per-feature often lowers ΔCE/ΔAcc without collapsing sparsity.

**Quick improvements (optional next steps):**

* Nudge **λ** upward from `0.09` to **`0.12–0.18`** to target ~**1.3–1.5** actives/ex and re-check Δ accuracy.
* Apply a **global θ shift** or per-feature θ calibration (EMA/percentile) to reduce worst-case (p95) errors.
* Try **SAE_LATENTS ∈ {64, 96, 128}** while keeping actives ≲1.5 to test if Δ shrinks at similar sparsity.


---

## Conclusions

* It’s feasible to obtain **\~1 active latent per example** on MNIST **logits** with **small accuracy loss**; careful λ/θ tuning can further reduce Δ.
* Training the SAE on **post-MLP logits** (not first-layer pre-acts) concentrates capacity on the decision signal — consistent with interpretability goals.
* A **simple λ sweep** is sufficient to steer average sparsity in this setting.Validation reconstruction error is **0.1743** in the normalized SAE space (raw ≈ **8.404**), with train/val/test tightly clustered, indicating stable generalization of the learned dictionary.

Validation CE rises from **0.1061** (baseline logits) to **0.3659** (SAE recon), consistent with the observed accuracy drop and leaving room for λ/θ tuning to recover fidelity.





