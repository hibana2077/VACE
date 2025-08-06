## B) Experiments to Run & Data to Collect

### B.1 Datasets & Splits

* **Ultra-FGVC benchmark subsets** (e.g., SoyLocal, SoyGlobal, SoyAgeing, SoyGene, Cotton80) with official splits.
* If desired, add a conventional fine-grained dataset for diversity (e.g., CUB-200-2011) to show generality.
* Use the **same augmentations** across baselines and VACE; report train/val/test splits clearly.

### B.2 Backbones (timm CNNs)

* Small set for coverage and runtime:

  * **ResNet-50** (classic baseline)
  * **ConvNeXt-Tiny** (modern CNN)
  * **EfficientNetV2-S** (parameter-efficient)
* (Optional) One larger backbone (**ResNet-101** or **ConvNeXt-S**) to test scaling.

### B.3 Baselines & Variants (Supervised only)

* **CE** (vanilla cross-entropy)
* **CE + Label Smoothing** (e.g., ε = 0.1)
* **LDAM** (for margin-based baseline)
* **VACE (ours)** default: `a=1.0, b in {0.5, 1.0}, tau_min=0.5, tau_max=2.0, ema_decay=0.1`
* **Ablations of VACE**

  1. **No EMA**: per-batch statistics only.
  2. **Different τ ranges**: tau\_min ∈ {0.5, 0.75}, tau\_max ∈ {1.5, 2.0, 3.0}.
  3. **b = 0** (degenerate → CE with constant τ=a).
  4. **Variance source**: (i) self-logit variance (default), (ii) feature variance (global pooled feature).
  5. **With/without label smoothing** (to test compatibility and calibration effects).
  6. **EMA decay**: ρ ∈ {0.05, 0.1, 0.3}.

### B.4 Training Regimes & Hyperparameters

* **Input size**: e.g., 224×224 (and 288×288 for a higher-res check).
* **Batch size**: choose per GPU memory (e.g., 64 on 24GB; adjust accordingly).
* **Epochs**: 100 (main), 200 (long-run stability check).
* **Optimizer**:

  * **SGD** (mom=0.9) with cosine decay, or **AdamW** with cosine & warmup (5 epochs).
* **LR**: grid over {0.1, 0.05} for SGD or {3e-4, 1e-4} for AdamW.
* **WD**: {1e-4, 5e-5}.
* **AMP**: enabled.
* **Dropout/Stochastic Depth**: off by default (keep consistent across methods unless being ablated).
* **Seeds**: at least 3 per configuration for mean±std reporting.

### B.5 Effectiveness Metrics to Collect

* **Top-1 accuracy** (primary), **Top-5 accuracy** (if K ≥ 5).
* **Per-class metrics**: macro **F1**, macro **Recall** (sensitive to ultra-fine categories).
* **Confusion matrix** (to show class confusion structure).
* **Margin statistics**: `margin = (z_y - max_{k≠y} z_k) / τ_y` distribution (mean, std).
* **Learning curves**: train/val loss & accuracy vs. epoch.

### B.6 Calibration Metrics to Collect

* **ECE** (Expected Calibration Error) with 15–20 bins.
* **NLL** (negative log-likelihood / cross-entropy on z/τ).
* **Brier score** (multi-class version).
* **Reliability diagrams** per method.

### B.7 Efficiency & Resource Metrics

* **Throughput**: images/sec (train, eval) under identical hardware and AMP.
* **Wall-clock time per epoch**.
* **Peak GPU memory (MiB)**.
* **Extra parameters & buffers** introduced by VACE (report O(K) τ and stats; numeric counts).
* **Overhead vs. CE**: Δtime/epoch, Δpeak VRAM, ΔFLOPs (should be \~0), all %.

### B.8 Ablation Studies (Recommended Minimum Set)

1. **Effect of b (slope)**: b ∈ {0, 0.25, 0.5, 1.0, 2.0} with fixed a=1.
2. **τ clipping range**: compare (0.5, 1.5) vs (0.5, 2.0) vs (0.75, 2.0).
3. **EMA decay**: ρ ∈ {0.05, 0.1, 0.3}.
4. **Variance source**: self-logit vs. feature variance.
5. **With/without label smoothing**: check ECE and NLL.
6. **Backbone sensitivity**: ResNet-50 vs ConvNeXt-Tiny vs EfficientNetV2-S.

### B.9 Reporting & Statistical Rigor

* Report **mean ± std** across seeds (n≥3).
* Include **confidence intervals** (e.g., 95% via t-distribution) for primary metrics.
* Perform **paired statistical tests** (e.g., paired t-test across seeds) comparing VACE vs CE.
* When multiple datasets/backbones, consider a **sign test** across tasks to show consistency.

### B.10 Reproducibility Package

* **Config files** (YAML/JSON) for every experiment variant.
* **Exact `timm` model names** and versions; log PyTorch/CUDA versions.
* **Deterministic flags**: seed everything; note any non-determinism due to CuDNN.
* **Checkpoints**: best-by-val (and last epoch) with τ buffers stored.
* **Scripts**: `train.py`, `eval.py`, `plot_calibration.py` (ECE & reliability), `profile_runtime.py`.
* **Logging**: TensorBoard or W\&B: store metrics, τ trajectories, memory, throughput.

---

### Minimal Pseudocode Snippet (for clarity)

```python
model = timm.create_model(name, pretrained=True, num_classes=0)  # remove head
head = nn.Linear(model.num_features, K)
loss_fn = VACE(num_classes=K, a=1.0, b=1.0, tau_min=0.5, tau_max=2.0, ema_decay=0.1)

for epoch in range(E):
    model.train(); head.train(); loss_fn.train()
    for x, y in train_loader:
        with torch.autocast(device_type='cuda', enabled=True):
            f = model.forward_features(x)
            if f.dim() > 2: f = f.mean((2,3))
            z = head(f)
            loss = loss_fn(z, y)  # updates EMAStats internally when training
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    # Validation
    model.eval(); head.eval(); loss_fn.eval()
    with torch.no_grad():
        for x, y in val_loader:
            f = model.forward_features(x)
            if f.dim() > 2: f = f.mean((2,3))
            z = head(f)
            tau = loss_fn.get_tau()  # fixed τ
            probs = (z / tau).softmax(dim=1)
            # collect metrics (Top-1, ECE, NLL, etc.)
```

---

**Deliverables Checklist for the Paper**

* Plots: accuracy bars (per dataset/backbone), reliability diagrams, τ histograms over epochs, margin distributions.
* Tables: mean±std Top-1, ECE, NLL, Brier, runtime/VRAM overhead vs CE; ablation tables for b, τ range, EMA decay.
* Appendix: additional confusion matrices, per-class results, and learning curves.

