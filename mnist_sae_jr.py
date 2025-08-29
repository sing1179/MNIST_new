
import os, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision import datasets, transforms
import pytorch_lightning as pl

SEED = 42
pl.seed_everything(SEED, workers=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Base classifier
FFN_TYPE     = "ReLU"   # "ReLU" or "GeGLU"
HIDDEN_DIM   = 128
LR_FFN       = 1e-3
EPOCHS_FFN   = 3
BATCH_FFN    = 8         # reviewer asked for small batch size

# SAE/Buffer
NORMALIZE_Y  = True      # standardize logits per-dim
SAE_LATENTS  = 32        # logits are 10-d; 32 latents is plenty
TAU          = 0.1       # STE temperature
INIT_THETA   = 0.5
LR_SAE_FINAL = 1e-3
EPOCHS_SAE_FINAL = 5
BATCH_SAE    = 8
TARGET_ACTIVES = [1]
OUT_DIR = "artifacts_mnist_sae_logits"
os.makedirs(OUT_DIR, exist_ok=True)

def load_mnist(batch_size=BATCH_FFN):
    tfm = transforms.Compose([transforms.ToTensor()])
    ds_train_full = datasets.MNIST(root=".", train=True,  download=True, transform=tfm)
    ds_test       = datasets.MNIST(root=".", train=False, download=True, transform=tfm)

    train_len  = int(0.8 * len(ds_train_full))
    val_len    = len(ds_train_full) - train_len
    ds_train, ds_val = random_split(ds_train_full, [train_len, val_len], generator=torch.Generator().manual_seed(SEED))

    pin = torch.cuda.is_available()
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=pin)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin)
    dl_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin)
    return dl_train, dl_val, dl_test

dl_train, dl_val, dl_test = load_mnist()

class FFN_GeGLU(nn.Module):
    def __init__(self, d_i, d_h, d_o):
        super().__init__()
        self.W_in_ih   = nn.Parameter(torch.randn(d_i, d_h) * 0.02)
        self.W_gate_ih = nn.Parameter(torch.randn(d_i, d_h) * 0.02)
        self.W_out_ho  = nn.Parameter(torch.randn(d_h, d_o) * 0.02)
    def forward(self, x_bi):  # x_bi: [batch, input]
        x_proj_bh = torch.einsum('bi,ih->bh', x_bi, self.W_in_ih)
        gate_bh   = F.gelu(torch.einsum('bi,ih->bh', x_bi, self.W_gate_ih))
        h_bh = x_proj_bh * gate_bh
        y_bo = torch.einsum('bh,ho->bo', h_bh, self.W_out_ho)  # logits
        return y_bo

class FFN_ReLU(nn.Module):
    def __init__(self, d_i, d_h, d_o):
        super().__init__()
        self.W_in_ih  = nn.Parameter(torch.randn(d_i, d_h) * 0.02)
        self.W_out_ho = nn.Parameter(torch.randn(d_h, d_o) * 0.02)
    def forward(self, x_bi):
        z_bh = torch.einsum('bi,ih->bh', x_bi, self.W_in_ih)
        h_bh = F.relu(z_bh)
        y_bo = torch.einsum('bh,ho->bo', h_bh, self.W_out_ho)  # logits
        return y_bo

class MNIST_FFN(pl.LightningModule):
    def __init__(self, ffn_type="ReLU", d_h=128, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        d_i, d_o = 28*28, 10
        if ffn_type == "GeGLU":
            self.ffn = FFN_GeGLU(d_i, d_h, d_o)
        elif ffn_type == "ReLU":
            self.ffn = FFN_ReLU(d_i, d_h, d_o)
        else:
            raise ValueError("Invalid ffn_type")
    def forward(self, x_bchw):
        x_bi = x_bchw.view(x_bchw.size(0), -1)
        y_bo = self.ffn(x_bi)
        return y_bo
    def training_step(self, batch, _):
        x_bchw, y_gt = batch
        y_bo = self(x_bchw)
        return F.cross_entropy(y_bo, y_gt)
    def validation_step(self, batch, _):
        x_bchw, y_gt = batch
        y_bo = self(x_bchw)
        acc = (y_bo.argmax(dim=1) == y_gt).float().mean()
        self.log("val_acc", acc, prog_bar=True)
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

def _trainer(max_epochs):
    use_gpu = torch.cuda.is_available()
    has_bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    precision = "bf16-mixed" if (use_gpu and has_bf16) else (16 if use_gpu else 32)
    return pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        precision=precision,
        logger=False,
        enable_checkpointing=False,
    )

model = MNIST_FFN(FFN_TYPE, HIDDEN_DIM, LR_FFN)
_tr = _trainer(EPOCHS_FFN)
_tr.fit(model, dl_train, dl_val)

@torch.no_grad()
def baseline_accuracy(model, loader):
    model.eval().to(device)
    correct=total=0
    for x_bchw, y_gt in loader:
        y_bo = model(x_bchw.to(device))
        pred = y_bo.argmax(dim=1)
        correct += (pred == y_gt.to(device)).sum().item()
        total   += y_gt.numel()
    return correct / max(total,1)

base_test_acc = baseline_accuracy(model, dl_test)
print(f"Baseline test accuracy: {base_test_acc:.4f}")

@torch.no_grad()
def collect_logits_buffers(model, loader):
    model.eval().to(device)
    feats = []
    for x_bchw, _ in loader:
        x_bi = x_bchw.to(device).view(x_bchw.size(0), -1)
        y_bo = model.ffn(x_bi)
        feats.append(y_bo.cpu())
    return torch.cat(feats, dim=0)

Ytr_bo = collect_logits_buffers(model, dl_train)
Yva_bo = collect_logits_buffers(model, dl_val)
Yte_bo = collect_logits_buffers(model, dl_test)
print("Buffers (logits):", Ytr_bo.shape, Yva_bo.shape, Yte_bo.shape)

if NORMALIZE_Y:
    mu_bo  = Ytr_bo.mean(0, keepdim=True)
    std_bo = Ytr_bo.std(0, keepdim=True).clamp_min(1e-6)
    Ytr_n = (Ytr_bo - mu_bo) / std_bo
    Yva_n = (Yva_bo - mu_bo) / std_bo
    Yte_n = (Yte_bo - mu_bo) / std_bo
else:
    mu_bo, std_bo = 0.0, 1.0
    Ytr_n, Yva_n, Yte_n = Ytr_bo, Yva_bo, Yte_bo

class SAE_JumpReLU(pl.LightningModule):
    def __init__(self, d_in, d_latents=32, lambda_l0=1e-2, lr=1e-3, init_theta=0.5, tau=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.enc = nn.Linear(d_in, d_latents, bias=True)
        self.theta_h = nn.Parameter(torch.full((d_latents,), float(init_theta)))
        self.tau   = tau
        self.dec = nn.Linear(d_latents, d_in, bias=True)
    def forward(self, y_bO):
        u_bh = self.enc(y_bO)
        soft_bh = torch.sigmoid((u_bh - self.theta_h) / self.tau)
        hard_bh = (u_bh > self.theta_h).float()
        gate_bh = (hard_bh - soft_bh).detach() + soft_bh
        f_bh = u_bh * gate_bh
        y_hat_bO = self.dec(f_bh)
        return y_hat_bO, f_bh, u_bh, soft_bh, hard_bh
    def _step(self, batch):
        (y_bO,) = batch
        y_hat_bO, f_bh, u_bh, soft_bh, hard_bh = self(y_bO)
        recon = F.mse_loss(y_hat_bO, y_bO, reduction="mean")
        l0_soft = soft_bh.sum(dim=1).mean()
        loss = recon + self.hparams.lambda_l0 * l0_soft
        l0_hard = hard_bh.sum(dim=1).float().mean().detach()
        return loss, recon.detach(), l0_soft.detach(), l0_hard
    def training_step(self, batch, _):
        loss, recon, l0_soft, l0_hard = self._step(batch)
        self.log_dict({"train/recon_mse": recon, "train/l0_soft": l0_soft, "train/l0_hard": l0_hard}, prog_bar=True)
        return loss
    def validation_step(self, batch, _):
        loss, recon, l0_soft, l0_hard = self._step(batch)
        self.log_dict({"val/recon_mse": recon, "val/l0_soft": l0_soft, "val/l0_hard": l0_hard}, prog_bar=True)
    def on_after_backward(self):
        with torch.no_grad():
            W_hO = self.dec.weight.data
            norms = W_hO.norm(dim=0, keepdim=True).clamp_min(1e-8)
            self.dec.weight.data = W_hO / norms
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

def tloader(t, bs=BATCH_SAE, shuffle=True):
    pin = torch.cuda.is_available()
    return DataLoader(TensorDataset(t), batch_size=bs, shuffle=shuffle, num_workers=2, pin_memory=pin)

@torch.no_grad()
def gate_counts(sae, Y_bO, batch=1024):
    sae.eval().to(device)
    tots = []
    for i in range(0, Y_bO.size(0), batch):
        u_bh = sae.enc(Y_bO[i:i+batch].to(device))
        hard_bh = (u_bh > sae.theta_h).float()
        tots.append(hard_bh.sum(dim=1).cpu())
    return float(torch.cat(tots).mean())

def sae_forward_logits(sae, y_raw_bO, mu_bo, std_bo):
    dev = y_raw_bO.device
    sae.eval().to(dev)
    mu_t  = mu_bo.to(dev)  if isinstance(mu_bo,  torch.Tensor) else mu_bo
    std_t = std_bo.to(dev) if isinstance(std_bo, torch.Tensor) else std_bo
    y_n_bO = (y_raw_bO - mu_t) / std_t if isinstance(mu_t, torch.Tensor) else y_raw_bO
    y_hat_n_bO, *_ = sae(y_n_bO)
    if isinstance(mu_t, torch.Tensor):
        y_hat_bO = y_hat_n_bO * std_t + mu_t
    else:
        y_hat_bO = y_hat_n_bO
    return y_hat_bO

@torch.no_grad()
def test_accuracy_with_sae_logits(model, loader, sae, mu_bo, std_bo):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(dev); sae.eval().to(dev)
    correct = total = 0
    for x_bchw, y_gt in loader:
        x_bi = x_bchw.to(dev).view(x_bchw.size(0), -1)
        y_raw_bO = model.ffn(x_bi)
        y_hat_bO = sae_forward_logits(sae, y_raw_bO, mu_bo, std_bo)
        pred = y_hat_bO.argmax(dim=1)
        correct += (pred == y_gt.to(dev)).sum().item()
        total   += y_gt.numel()
    return correct / max(total, 1)

def _trainer_sae(max_epochs):
    use_gpu = torch.cuda.is_available()
    has_bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    precision = "bf16-mixed" if (use_gpu and has_bf16) else (16 if use_gpu else 32)
    return pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        precision=precision,
        logger=False,
        enable_checkpointing=False,
    )

def train_sae_once(d_in, lam, Ytr, Yva, epochs):
    sae = SAE_JumpReLU(d_in=d_in, d_latents=SAE_LATENTS, lambda_l0=lam, lr=LR_SAE_FINAL, init_theta=INIT_THETA, tau=TAU)
    tr = _trainer_sae(epochs)
    tr.fit(sae, tloader(Ytr, shuffle=True), tloader(Yva, shuffle=False))
    return sae

def calibrate_lambda(Ytr, Yva, target_actives, coarse_grid=np.geomspace(1e-6, 1e-1, 10), refine_factor=3, refine_steps=5):
    d_in = Ytr.shape[1]
    best = None
    for lam in coarse_grid:
        sae = train_sae_once(d_in, float(lam), Ytr, Yva, epochs=1)
        m_act = gate_counts(sae, Yva)
        gap = abs(m_act - target_actives)
        if (best is None) or (gap < best["gap"]):
            best = {"lam": float(lam), "sae": sae, "m_act": float(m_act), "gap": float(gap)}
    lam_star = best["lam"]
    low = lam_star / (refine_factor**2)
    high = lam_star * (refine_factor**2)
    refine_grid = np.geomspace(max(low, 1e-8), min(high, 1.0), refine_steps)
    for lam in refine_grid:
        sae = train_sae_once(d_in, float(lam), Ytr, Yva, epochs=1)
        m_act = gate_counts(sae, Yva)
        gap = abs(m_act - target_actives)
        if gap < best["gap"]:
            best = {"lam": float(lam), "sae": sae, "m_act": float(m_act), "gap": float(gap)}
    return best

# device-safe overrides that work with both SAE variants (with .theta or .act.theta)
def _get_theta(sae):
    return sae.theta if hasattr(sae, "theta") else sae.act.theta

def sae_forward_modes(sae, z_raw, mu, std, mode="jumprelu"):
    dev = z_raw.device
    sae.eval().to(dev)
    mu_t  = mu.to(dev)  if isinstance(mu,  torch.Tensor) else mu
    std_t = std.to(dev) if isinstance(std, torch.Tensor) else std
    z = (z_raw - mu_t) / std_t if isinstance(mu_t, torch.Tensor) else z_raw

    u = sae.enc(z)
    theta = _get_theta(sae)
    if mode == "jumprelu":
        b = (u > theta).float()
        f = u * b
    elif mode == "boolean":
        f = (u > theta).float()
    else:
        raise ValueError("mode must be 'jumprelu' or 'boolean'")

    xh = sae.dec(f)
    if isinstance(mu_t, torch.Tensor):
        xh = xh * std_t + mu_t
    return xh

@torch.no_grad()
def test_accuracy_with_mode(model, loader, ffn_type, sae, mu, std, mode="jumprelu"):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(dev)
    sae.eval().to(dev)

    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(dev), yb.to(dev)
        x_flat = xb.view(xb.size(0), -1)
        z_raw  = torch.einsum('bi,ih->bh', x_flat, model.ffn.W_in)
        z_hat  = sae_forward_modes(sae, z_raw, mu, std, mode=mode)

        if ffn_type == "ReLU":
            h = F.relu(z_hat)
            logits = torch.einsum('bh,ho->bo', h, model.ffn.W_out)
        else:
            gate = F.gelu(torch.einsum('bi,ih->bh', x_flat, model.ffn.W_gate))
            h = z_hat * gate
            logits = torch.einsum('bh,ho->bo', h, model.ffn.W_out)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total   += yb.numel()
    return correct / max(total, 1)

# Cell 9 — robust & fast SAE run (fixes Cell 9 crash)
# - Uses small subsets for calibration/final SAE
# - Small lambda grid + few refine steps
# - Try/except around long-running calls so interrupts don't explode the traceback

# Speed knobs
FAST_MODE = True
CAL_SAMPLES_TRAIN = 8000 if FAST_MODE else len(Ytr_n)
CAL_SAMPLES_VAL   = 2000 if FAST_MODE else len(Yva_n)
COARSE_GRID       = np.geomspace(1e-5, 1e-2, 4)  # tiny grid
REFINE_STEPS      = 3
FINAL_EPOCHS      = min(EPOCHS_SAE_FINAL, 3) if FAST_MODE else EPOCHS_SAE_FINAL

# Subsets for quick calibration/training
Ytr_cal = Ytr_n[:CAL_SAMPLES_TRAIN].contiguous()
Yva_cal = Yva_n[:CAL_SAMPLES_VAL].contiguous()
print(f"Calibrate on train={len(Ytr_cal)}, val={len(Yva_cal)}; coarse_grid={COARSE_GRID}")

results = []
for target_k in TARGET_ACTIVES:
    print(f"\n=== Calibrating for target actives ≈ {target_k} on LOGITS ===")
    # --- Calibration ---
    try:
        pick = calibrate_lambda(
            Ytr_cal, Yva_cal, target_k,
            coarse_grid=COARSE_GRID,
            refine_steps=REFINE_STEPS
        )
    except KeyboardInterrupt:
        print("Calibration interrupted — falling back to λ=1e-3")
        pick = {"lam": 1e-3, "m_act": float("nan")}
    print(f"Picked lambda={pick['lam']:.2e}; achieved actives ≈ {pick.get('m_act', float('nan')):.2f} (cal)")

    # --- Train final SAE ---
    try:
        sae_final = train_sae_once(Ytr_cal.shape[1], pick["lam"], Ytr_cal, Yva_cal, epochs=FINAL_EPOCHS)
    except KeyboardInterrupt:
        print("Final SAE training interrupted — stopping cleanly.")
        raise  # re-raise so you can stop the run without a messy stacktrace

    # Evaluate (use full val/test – cheap)
    achieved_k   = gate_counts(sae_final, Yva_n)
    acc_baseline = base_test_acc
    acc_sae      = test_accuracy_with_sae_logits(model, dl_test, sae_final, mu_bo, std_bo)

    row = {
        "target_actives": target_k,
        "achieved_actives_val": round(achieved_k, 2),
        "lambda": pick["lam"],
        "baseline_acc": round(acc_baseline, 4),
        "recon_acc_logits": round(acc_sae, 4),
        "delta_acc_logits": round(acc_baseline - acc_sae, 4),
        "sae_latents": SAE_LATENTS,
        "tau": TAU,
        "normalize_logits": NORMALIZE_Y,
        "batch_sizes": {"ffn": BATCH_FFN, "sae": BATCH_SAE},
    }
    results.append(row)
    print(row)

# Persist results
with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
    json.dump(results, f, indent=2)

# Pick & write the "best" row
best = min(results, key=lambda r: (abs(r["target_actives"]-1), r["delta_acc_logits"]))
best_md = (
    "# Best Result (Auto)\n"
    f"- target_actives: {best['target_actives']}\n"
    f"- achieved_actives_val: {best['achieved_actives_val']}\n"
    f"- lambda: {best['lambda']:.3e}\n"
    f"- baseline_acc: {best['baseline_acc']:.4f}\n"
    f"- recon_acc_logits: {best['recon_acc_logits']:.4f}\n"
    f"- delta_acc_logits: {best['delta_acc_logits']:.4f}\n"
    f"- sae_latents: {best['sae_latents']}\n"
    f"- tau: {best['tau']}\n"
    f"- normalize_logits: {best['normalize_logits']}\n"
    f"- batch_sizes: FFN={best['batch_sizes']['ffn']}, SAE={best['batch_sizes']['sae']}\n"
)
with open(os.path.join(OUT_DIR, "BEST_RESULTS.md"), "w") as f:
    f.write(best_md)

print("\nWrote:", os.path.join(OUT_DIR, "summary.json"))
print("Wrote:", os.path.join(OUT_DIR, "BEST_RESULTS.md"))

best_path = os.path.join(OUT_DIR, "BEST_RESULTS.md")
if os.path.exists(best_path):
    with open(best_path, "r") as f:
        print(f.read())
else:
    print("Run Cell 9 first to generate BEST_RESULTS.md")

# Cell 12 — Consistent reconstruction-MSE evaluators (normalized vs unnormalized)

import torch
import torch.nn.functional as F

@torch.no_grad()
def sae_forward_recon(sae, z_raw, mu=None, std=None, mode="jumprelu"):
    """
    Returns:
      z_hat_norm  : reconstruction in the *normalized* SAE space
      z_recon_raw : reconstruction mapped back to *raw* (unnormalized) space (if mu/std given)
    """
    dev = next(sae.parameters()).device
    z_raw = z_raw.to(dev)

    # normalize to the space used during SAE training
    if (isinstance(mu, torch.Tensor) and isinstance(std, torch.Tensor)):
        mu_t  = mu.to(dev)
        std_t = std.to(dev)
        z = (z_raw - mu_t) / (std_t + 1e-8)
    else:
        mu_t = std_t = None
        z = z_raw

    u = sae.enc(z)

    # --- SAFE theta retrieval (no tensor truthiness) ---
    theta = None
    if hasattr(sae, "theta_h"):
        theta = getattr(sae, "theta_h")
    elif hasattr(sae, "theta"):
        theta = getattr(sae, "theta")
    if isinstance(theta, torch.nn.Parameter):
        theta = theta.data
    if theta is None:
        theta = torch.tensor(0.0, device=dev, dtype=u.dtype)
    else:
        theta = theta.to(device=dev, dtype=u.dtype)
    # ---------------------------------------------------

    if mode == "jumprelu":
        b = (u > theta).to(u.dtype)
        f = u * b
    elif mode == "boolean":
        f = (u > theta).to(u.dtype)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    z_hat_norm = sae.dec(f)  # normalized space

    if (mu_t is not None) and (std_t is not None):
        z_recon_raw = z_hat_norm * (std_t + 1e-8) + mu_t
    else:
        z_recon_raw = z_hat_norm

    return z_hat_norm, z_recon_raw


@torch.no_grad()
def mse_over_loader(sae, loader, mu=None, std=None, mode="jumprelu", compare_space="normalized"):
    """
    compare_space: "normalized"  -> MSE between z_hat_norm and z_norm   (matches SAE training logs)
                   "raw"         -> MSE between z_recon_raw and z_raw   (user-facing, intuitive)
    Returns scalar mean MSE over all examples and dimensions.
    """
    sae.eval()
    dev = next(sae.parameters()).device
    total_sqerr, total_count = 0.0, 0

    for batch in loader:
        # accept (tensor,) or tensor
        z_raw = batch[0] if isinstance(batch, (list, tuple)) else batch
        z_raw = z_raw.to(dev)

        # forward
        z_hat_norm, z_recon_raw = sae_forward_recon(sae, z_raw, mu=mu, std=std, mode=mode)

        if compare_space == "normalized":
            if (isinstance(mu, torch.Tensor) and isinstance(std, torch.Tensor)):
                z_norm = (z_raw - mu.to(dev)) / (std.to(dev) + 1e-8)
            else:
                z_norm = z_raw
            diff = z_hat_norm - z_norm
        elif compare_space == "raw":
            diff = z_recon_raw - z_raw
        else:
            raise ValueError("compare_space must be 'normalized' or 'raw'")

        total_sqerr += diff.pow(2).sum().item()
        total_count += diff.numel()

    return total_sqerr / max(total_count, 1)

# Cell 13 — Compute train/val/test MSE in normalized space (matches training logs) and raw space

# Reuse your existing hidden-activation tensors & loaders (the same reps you trained the SAE on):
# Assuming you already have Ytr_bo, Yva_bo, Yte_bo and mu_bo, std_bo, sae_final
train_loader = tloader(Ytr_bo, bs=256, shuffle=False)
val_loader   = tloader(Yva_bo, bs=256, shuffle=False)
test_loader  = tloader(Yte_bo, bs=256, shuffle=False)

# 1) Normalized-space MSE (should be close to Lightning's train/val recon_mse ~ 0.2)
tr_mse_norm = mse_over_loader(sae_final, train_loader, mu=mu_bo, std=std_bo, compare_space="normalized")
va_mse_norm = mse_over_loader(sae_final, val_loader,   mu=mu_bo, std=std_bo, compare_space="normalized")
te_mse_norm = mse_over_loader(sae_final, test_loader,  mu=mu_bo, std=std_bo, compare_space="normalized")

# 2) Raw-space MSE (often larger because it includes the original scale)
tr_mse_raw = mse_over_loader(sae_final, train_loader, mu=mu_bo, std=std_bo, compare_space="raw")
va_mse_raw = mse_over_loader(sae_final, val_loader,   mu=mu_bo, std=std_bo, compare_space="raw")
te_mse_raw = mse_over_loader(sae_final, test_loader,  mu=mu_bo, std=std_bo, compare_space="raw")

print(f"[Recon MSE — normalized space] train={tr_mse_norm:.6f} | val={va_mse_norm:.6f} | test={te_mse_norm:.6f}")
print(f"[Recon MSE — raw space]        train={tr_mse_raw:.6f}  | val={va_mse_raw:.6f}  | test={te_mse_raw:.6f}")

# Cell 14-per-example MSE distribution and best/worst examples (on validation set)

import torch

@torch.no_grad()
def per_example_mse(sae, Z_raw, mu=None, std=None, mode="jumprelu", compare_space="normalized", batch_size=512):
    dev = next(sae.parameters()).device
    sae.eval()
    N = Z_raw.shape[0]
    out = torch.empty(N, device="cpu")
    for i in range(0, N, batch_size):
        z_batch = Z_raw[i:i+batch_size].to(dev)
        z_hat_norm, z_recon_raw = sae_forward_recon(sae, z_batch, mu=mu, std=std, mode=mode)
        if compare_space == "normalized":
            z_norm = (z_batch - mu.to(dev)) / (std.to(dev) + 1e-8) if (isinstance(mu, torch.Tensor) and isinstance(std, torch.Tensor)) else z_batch
            diff = z_hat_norm - z_norm
        else:
            diff = z_recon_raw - z_batch
        out[i:i+batch_size] = diff.pow(2).mean(dim=1).detach().cpu()
    return out  # (N,)

# Compute per-example val MSE in normalized space
val_per_ex_mse = per_example_mse(sae_final, Yva_bo, mu=mu_bo, std=std_bo, compare_space="normalized")
val_avg = val_per_ex_mse.mean().item()
val_med = val_per_ex_mse.median().item()
val_p95 = val_per_ex_mse.quantile(0.95).item()
best_idx = int(torch.argmin(val_per_ex_mse))
worst_idx = int(torch.argmax(val_per_ex_mse))

print(f"[Val per-example MSE — normalized] mean={val_avg:.6f} | median={val_med:.6f} | p95={val_p95:.6f}")
print(f"Best example idx={best_idx} mse={val_per_ex_mse[best_idx].item():.6f}")
print(f"Worst example idx={worst_idx} mse={val_per_ex_mse[worst_idx].item():.6f}")
# Cell 15 — CE/accuracy helpers that work with your loaders and model

import torch
import torch.nn.functional as F

@torch.no_grad()
def ce_and_acc_over_img_loader(model, loader, device=None):
    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model.eval().to(dev)
    ce_sum, correct, seen = 0.0, 0, 0
    for x_bchw, y_gt in loader:
        x_bchw = x_bchw.to(dev)
        y_gt   = y_gt.to(dev).long()
        logits = model(x_bchw)                # baseline logits
        ce_sum += F.cross_entropy(logits, y_gt, reduction="sum").item()
        correct += (logits.argmax(dim=1) == y_gt).sum().item()
        seen += y_gt.numel()
    return ce_sum / max(seen, 1), correct / max(seen, 1)

@torch.no_grad()
def ce_and_acc_over_img_loader_with_sae(model, loader, sae, mu_bo, std_bo, device=None):
    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model.eval().to(dev); sae.eval().to(dev)
    ce_sum, correct, seen = 0.0, 0, 0
    for x_bchw, y_gt in loader:
        x_bchw = x_bchw.to(dev)
        y_gt   = y_gt.to(dev).long()
        # your pipeline: x -> MLP.ffn logits -> SAE -> recon logits
        x_bi      = x_bchw.view(x_bchw.size(0), -1)
        y_raw_bO  = model.ffn(x_bi)
        y_hat_bO  = sae_forward_logits(sae, y_raw_bO, mu_bo, std_bo)  # already returns RAW-space recon logits
        ce_sum   += F.cross_entropy(y_hat_bO, y_gt, reduction="sum").item()
        correct  += (y_hat_bO.argmax(dim=1) == y_gt).sum().item()
        seen     += y_gt.numel()
    return ce_sum / max(seen, 1), correct / max(seen, 1)
# Cell 16 — Compute CE/Acc for baseline vs SAE on all splits

# Baseline (no SAE)
tr_ce_base, tr_acc_base = ce_and_acc_over_img_loader(model, dl_train, device=device)
va_ce_base, va_acc_base = ce_and_acc_over_img_loader(model, dl_val,   device=device)
te_ce_base, te_acc_base = ce_and_acc_over_img_loader(model, dl_test,  device=device)

# With SAE-reconstructed logits
tr_ce_sae, tr_acc_sae = ce_and_acc_over_img_loader_with_sae(model, dl_train, sae_final, mu_bo, std_bo, device=device)
va_ce_sae, va_acc_sae = ce_and_acc_over_img_loader_with_sae(model, dl_val,   sae_final, mu_bo, std_bo, device=device)
te_ce_sae, te_acc_sae = ce_and_acc_over_img_loader_with_sae(model, dl_test,  sae_final, mu_bo, std_bo, device=device)

print("=== Cross-Entropy (↓) / Accuracy (↑) ===")
print(f"TRAIN | CE base {tr_ce_base:.4f} | CE SAE {tr_ce_sae:.4f} | ΔCE {tr_ce_sae - tr_ce_base:+.4f} | Acc base {tr_acc_base:.4f} | Acc SAE {tr_acc_sae:.4f} | ΔAcc {tr_acc_sae - tr_acc_base:+.4f}")
print(f"VAL   | CE base {va_ce_base:.4f} | CE SAE {va_ce_sae:.4f} | ΔCE {va_ce_sae - va_ce_base:+.4f} | Acc base {va_acc_base:.4f} | Acc SAE {va_acc_sae:.4f} | ΔAcc {va_acc_sae - va_acc_base:+.4f}")
print(f"TEST  | CE base {te_ce_base:.4f} | CE SAE {te_ce_sae:.4f} | ΔCE {te_ce_sae - te_ce_base:+.4f} | Acc base {te_acc_base:.4f} | Acc SAE {te_acc_sae:.4f} | ΔAcc {te_acc_sae - te_acc_base:+.4f}")
