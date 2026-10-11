# Verified CondiQuant checkpoints

Nine SwinIR-light checkpoints were evaluated on all five complete benchmarks (328 images per checkpoint). PSNR and SSIM use the Y channel, a scale-sized crop border, full-image inference, and no self-ensemble.

A pass allows at most a 0.05 dB PSNR deficit and 0.0005 SSIM deficit on each dataset. The exact measurements and paper deltas are available in [measured_results.csv](measured_results.csv) and the linked JSON reports.

All checkpoints use the latest weights after 4,000 iterations with λ=0.003, η=0.01, and K=50. Of the 45 PSNR measurements, 43 meet or exceed the paper; the two lower values have deficits of 0.017089 dB (×2 W3A3, Set5) and 0.014953 dB (×2 W4A4, Set5). All 45 SSIM measurements meet or exceed the paper values.

| Scale | Bits | Set5 | Set14 | B100 | Urban100 | Manga109 | Report |
|---|---|---|---|---|---|---|---|
| ×2 | W2A2 | 37.1877 / 0.956803 | 32.7992 / 0.910493 | 31.5748 / 0.891365 | 30.0714 / 0.905830 | 36.8854 / 0.971942 | [JSON](x2_w2a2.json) |
| ×2 | W3A3 | 37.7529 / 0.959411 | 33.2901 / 0.915499 | 31.9914 / 0.897154 | 31.3923 / 0.921633 | 38.0771 / 0.975674 | [JSON](x2_w3a3.json) |
| ×2 | W4A4 | 38.0150 / 0.960522 | 33.5656 / 0.918291 | 32.1853 / 0.899684 | 32.1693 / 0.929154 | 38.6857 / 0.977115 | [JSON](x2_w4a4.json) |
| ×3 | W2A2 | 33.0918 / 0.913567 | 29.4930 / 0.826060 | 28.4822 / 0.788654 | 26.4358 / 0.809952 | 31.0613 / 0.921856 | [JSON](x3_w2a2.json) |
| ×3 | W3A3 | 33.9577 / 0.922863 | 30.0584 / 0.837217 | 28.8776 / 0.799006 | 27.4833 / 0.837806 | 32.6577 / 0.937762 | [JSON](x3_w3a3.json) |
| ×3 | W4A4 | 34.3525 / 0.926431 | 30.3119 / 0.842131 | 29.0722 / 0.804223 | 28.1461 / 0.852803 | 33.3719 / 0.943868 | [JSON](x3_w4a4.json) |
| ×4 | W2A2 | 30.7427 / 0.869251 | 27.6466 / 0.758314 | 26.9547 / 0.714404 | 24.6126 / 0.731826 | 27.9031 / 0.866268 | [JSON](x4_w2a2.json) |
| ×4 | W3A3 | 31.6997 / 0.886888 | 28.2474 / 0.772900 | 27.3476 / 0.728434 | 25.4841 / 0.766684 | 29.4933 / 0.894842 | [JSON](x4_w3a3.json) |
| ×4 | W4A4 | 32.1281 / 0.893317 | 28.5619 / 0.781223 | 27.5448 / 0.735494 | 26.0506 / 0.786510 | 30.3004 / 0.907181 | [JSON](x4_w4a4.json) |

Checkpoints are floating-point fake-quantization files. The upload bundle also contains calibration tensors, effective training YAML files, and a SHA-256 manifest.

Full-precision protocol checks: [×2](fp_x2.json), [×3](fp_x3.json), [×4](fp_x4.json).
