# Reproducing CondiQuant

The main reproduction target is Table 3 of the [ICML 2026 paper](https://proceedings.mlr.press/v306/liu26eh.html): SwinIR-light, three scales, three bit-widths, and five benchmarks. `paper_results.csv` contains the 45 published CondiQuant PSNR/SSIM pairs and 15 full-precision reference pairs.

This checkout implements the supplied SwinIR code. It does not currently provide the EDSR or MambaIRv2 implementations from the paper's separate architecture-generalization experiment (Table 4).

## Protocol

1. Start from the official pretrained SwinIR-light model for the desired scale.
2. Prepare a deterministic 100-image DF2K subset and 100 RGB calibration patches with seed 3407.
3. Apply the SVD proximal-gradient procedure to attention projections. The original implementation excluded the 120-wide SwinIR-light FFN matrices; this release expresses that scope as `exclude_mlp: true` instead of a hardcoded shape check.
4. Initialize bounds with DOBI. Sample 64×64 LR patches from the selected images using the LR-only cached dataset; HR targets are not needed for distillation. Train only quantizer bounds and activation biases against the frozen full-precision teacher. Backbone weights and bit-widths remain fixed during this distillation stage.
5. Save the checkpoint, reload it in a fresh model, and evaluate all benchmark images. Compute PSNR/SSIM on the Y channel, with a scale-sized crop border and no self-ensemble.
6. Compare every metric with its paper reference. Keep the training configuration, calibration file, benchmark report, and SHA-256 checksum with the checkpoint.

The experimental-settings paragraph gives λ=0.003 and η=0.01; those are the YAML defaults. The supplied original source had λ=0.004 and η=0.001. Both parameters are configurable so this discrepancy can be assessed explicitly during reproduction.

Both settings were trained for 4,000 iterations at ×2 W4A4 with the same data and seed, and both passed all five benchmarks under the stated tolerance. The [paper-default report](verification/x2_w4a4.json) gives Set5 38.0150 / 0.960522 and Manga109 38.6857 / 0.977115. The [original-source settings report](verification/source_x2_w4a4.json) gives Set5 38.0328 / 0.960429 and Manga109 38.6182 / 0.976993. The nine-checkpoint upload bundle consistently uses the paper-default parameters.

## Changes made while preparing the original source

- Renamed the registered `TDQuantModel` to `CondiQuantModel` and added consistent ×2/×3/×4 configurations.
- Removed hardcoded CUDA allocation and preserved the scheduler's `CUDA_VISIBLE_DEVICES` assignment.
- Corrected identical lower/upper initial bounds, guarded zero quantization intervals, and corrected 4D DOBI channel indexing.
- Fixed feature-normalization broadcasting and froze the full-precision teacher and backbone during bound optimization.
- Rejected checkpoint/config bit-width mismatches and skipped SVD while initializing a trained checkpoint for loading.
- Added deterministic RGB calibration preparation and machine-readable evaluation reports.
- Count actual complete training batches when planning epochs, so training reaches the requested iteration count. Autograd anomaly tracing is opt-in (`train.detect_anomaly: true`).

These corrections require fresh numerical verification. Passing smoke tests alone does not establish reproduction of the paper's reported accuracy.

## Checks

```bash
python -m unittest discover -s tests -v
python -m basicsr.train --help
python -m basicsr.test --help
```

The integration checks cover low-bit forward/backward operations, constant inputs, SVD behavior, training a small SwinIR model, exact checkpoint round-tripping, rejection of a mismatched bit-width, complete training budgets, effective configuration snapshots, and upload-bundle integrity.

The verification script exits with code 2 if any dataset falls below the paper result by more than the configured tolerance. The default tolerances (0.05 dB and 0.0005 SSIM) are explicit release criteria, not proof of exact numerical equality.
