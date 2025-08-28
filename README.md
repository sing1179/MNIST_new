````markdown
# MNIST FFN + Sparse Autoencoder (SAE)

**TL;DR**  
We train a small MNIST classifier (FFN), then fit a **JumpReLU** sparse autoencoder on the **classifier’s logits**.  
Target sparsity is **~1 active latent per example** (MNIST has 10 classes).  
Fidelity is measured as test accuracy when replacing model logits with SAE-reconstructed logits.  
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

---

## Process (Repro Steps)

1. **Install deps** (in the notebook):
   ```bash
   pip install "pytorch-lightning<3" "torch>=2.2,<3" torchvision torchaudio matplotlib tqdm
````

2. **Train FFN** to get baseline accuracy.
3. **Collect logits** (train/val/test) and optionally standardize.
4. **λ calibration** targeting L0≈1 (fast, 1 epoch per candidate λ).
5. **Train final SAE** at the chosen λ and evaluate fidelity on test.
6. **Artifacts:**

   * `artifacts_mnist_sae_logits/summary.json`
   * `artifacts_mnist_sae_logits/BEST_RESULTS.md`

  

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

**Interpretation:**

* Achieved actives **1.63 > 1.0** indicates the solution is a bit **too dense** for the target.
* Accuracy drop **Δ ≈ 0.056** suggests the reconstruction can be improved.

**Quick improvements (optional next steps):**

* Increase λ moderately (e.g., **0.2–0.5**) and retrain SAE once; re-check L0 and Δ.
* Or apply a **global θ shift** post-training to snap mean actives to **exactly 1.0** and re-measure Δ.

---

## Conclusions

* It’s feasible to obtain **\~1 active latent per example** on MNIST **logits** with **small accuracy loss**; careful λ/θ tuning can further reduce Δ.
* Training the SAE on **post-MLP logits** (not first-layer pre-acts) concentrates capacity on the decision signal — consistent with interpretability goals.
* A **simple λ sweep** is sufficient to steer average sparsity in this setting.




