# Dataset layout

```text
datasets/
├── DF2K/
│   ├── HR/
│   └── LR_bicubic/
│       ├── X2/  # e.g. 0001x2.png for HR/0001.png
│       ├── X3/
│       └── X4/
└── benchmark/
    ├── Set5/
    ├── Set14/
    ├── B100/
    ├── Urban100/
    └── Manga109/
```

Each benchmark contains `HR/` and `LR_bicubic/X{2,3,4}/`. Default LR naming is `{HR_stem}x{scale}.png`; Manga109 uses `{HR_stem}_LRBI_x{scale}.png`. Adjust `filename_tmpl` in the YAML configuration if your files use different names.

The verification script requires the full benchmark sizes: Set5=5, Set14=14, B100=100, Urban100=100, Manga109=109. Evaluation computes PSNR and SSIM on the Y channel and crops a border equal to the upscale factor. Self-ensemble is disabled.

Dataset files are excluded from Git. Download and use them according to their respective distribution terms.
