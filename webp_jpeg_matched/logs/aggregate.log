# Matched-budget WebP/JPEG rerun (protocol WEBP_JPEG_RERUN_PROTOCOL.md) vs AquaSC

Test split, 432 images x 10 draws per cell; all records scored (capacity/length/CRC/decode failures -> black image). CI = paired bootstrap over images (AquaSC - baseline), per-image AP50 averaged over draws. AquaSC = Gray QPSK JointFrame (headline / retrained r061 / retrained r030); baselines = Gray 16-QAM, BG2 k=1024 rate-1/2 blocks.


## Budget low: 0.030612 uses/px, 1536 symbols/image


### WEBP q=1 (quality=q, method=4 (PIL defaults otherwise)); capacity 368 B; test capacity failures 25/432 (5.79%); clean-channel AP50 0.0763, PSNR 33.55, SSIM 0.8924

| SNR | base AP50 | base pooled mAP50 | base PSNR | base SSIM | black % | AquaSC AP50 | AquaSC PSNR | diff AP50 [95% CI] |
|---|---|---|---|---|---|---|---|---|
| -5 | 0.0008 | 0.0000 | 4.52 | 0.0027 | 100.0 | 0.3237 | 29.47 | +0.3229 [+0.3079, +0.3388] |
| +0 | 0.0017 | 0.0049 | 4.91 | 0.0154 | 98.7 | 0.4312 | 32.79 | +0.4295 [+0.4101, +0.4492] |
| +5 | 0.0207 | 0.0130 | 12.24 | 0.2440 | 74.4 | 0.4895 | 34.63 | +0.4687 [+0.4466, +0.4912] |
| +10 | 0.0487 | 0.0228 | 23.36 | 0.5834 | 38.5 | 0.5143 | 35.41 | +0.4656 [+0.4412, +0.4900] |
| +15 | 0.0677 | 0.0324 | 30.07 | 0.7875 | 16.9 | 0.5240 | 35.74 | +0.4563 [+0.4308, +0.4819] |
| +20 | 0.0726 | 0.0338 | 32.30 | 0.8545 | 9.8 | 0.5259 | 35.78 | +0.4533 [+0.4274, +0.4794] |

Actual transmitted symbols/image (min/mean/max over all records): 1536/1536.0/1536; AquaSC 1536.


### JPEG q=1 (quality=q, optimize=True (PIL default 4:2:0). optimize=True chosen on validation only: with PIL defaults the floor is ~1440 B/file, so low and mid budgets are 100% infeasible); capacity 368 B; test capacity failures 432/432 (100.00%); clean-channel AP50 0.0008, PSNR 4.52, SSIM 0.0027

| SNR | base AP50 | base pooled mAP50 | base PSNR | base SSIM | black % | AquaSC AP50 | AquaSC PSNR | diff AP50 [95% CI] |
|---|---|---|---|---|---|---|---|---|
| -5 | 0.0008 | 0.0000 | 4.52 | 0.0027 | 100.0 | 0.3237 | 29.47 | +0.3229 [+0.3079, +0.3388] |
| +0 | 0.0008 | 0.0000 | 4.52 | 0.0027 | 100.0 | 0.4312 | 32.79 | +0.4304 [+0.4110, +0.4501] |
| +5 | 0.0008 | 0.0000 | 4.52 | 0.0027 | 100.0 | 0.4895 | 34.63 | +0.4887 [+0.4668, +0.5109] |
| +10 | 0.0008 | 0.0000 | 4.52 | 0.0027 | 100.0 | 0.5143 | 35.41 | +0.5135 [+0.4898, +0.5371] |
| +15 | 0.0008 | 0.0000 | 4.52 | 0.0027 | 100.0 | 0.5240 | 35.74 | +0.5232 [+0.4994, +0.5473] |
| +20 | 0.0008 | 0.0000 | 4.52 | 0.0027 | 100.0 | 0.5259 | 35.78 | +0.5251 [+0.5011, +0.5493] |

Actual transmitted symbols/image (min/mean/max over all records): 1536/1536.0/1536; AquaSC 1536.


## Budget mid: 0.061224 uses/px, 3072 symbols/image


### WEBP q=9 (quality=q, method=4 (PIL defaults otherwise)); capacity 740 B; test capacity failures 1/432 (0.23%); clean-channel AP50 0.1385, PSNR 36.83, SSIM 0.9504

| SNR | base AP50 | base pooled mAP50 | base PSNR | base SSIM | black % | AquaSC AP50 | AquaSC PSNR | diff AP50 [95% CI] |
|---|---|---|---|---|---|---|---|---|
| -5 | 0.0008 | 0.0000 | 4.52 | 0.0027 | 100.0 | 0.3647 | 32.07 | +0.3639 [+0.3496, +0.3792] |
| +0 | 0.0035 | 0.0086 | 4.96 | 0.0163 | 98.6 | 0.5381 | 36.25 | +0.5346 [+0.5176, +0.5523] |
| +5 | 0.0381 | 0.0290 | 13.03 | 0.2566 | 73.2 | 0.6309 | 38.52 | +0.5928 [+0.5736, +0.6122] |
| +10 | 0.0888 | 0.0612 | 25.33 | 0.6163 | 35.4 | 0.6690 | 39.46 | +0.5802 [+0.5587, +0.6016] |
| +15 | 0.1214 | 0.0818 | 32.83 | 0.8346 | 12.4 | 0.6837 | 39.84 | +0.5623 [+0.5397, +0.5845] |
| +20 | 0.1321 | 0.0885 | 35.45 | 0.9104 | 4.4 | 0.6870 | 39.91 | +0.5548 [+0.5315, +0.5776] |

Actual transmitted symbols/image (min/mean/max over all records): 3072/3072.0/3072; AquaSC 3072.


### JPEG q=5 (quality=q, optimize=True (PIL default 4:2:0). optimize=True chosen on validation only: with PIL defaults the floor is ~1440 B/file, so low and mid budgets are 100% infeasible); capacity 740 B; test capacity failures 0/432 (0.00%); clean-channel AP50 0.0950, PSNR 28.27, SSIM 0.8666

| SNR | base AP50 | base pooled mAP50 | base PSNR | base SSIM | black % | AquaSC AP50 | AquaSC PSNR | diff AP50 [95% CI] |
|---|---|---|---|---|---|---|---|---|
| -5 | 0.0008 | 0.0000 | 4.52 | 0.0027 | 100.0 | 0.3647 | 32.07 | +0.3639 [+0.3496, +0.3792] |
| +0 | 0.0018 | 0.0035 | 4.70 | 0.0097 | 99.2 | 0.5381 | 36.25 | +0.5363 [+0.5191, +0.5539] |
| +5 | 0.0203 | 0.0100 | 9.72 | 0.1954 | 77.7 | 0.6309 | 38.52 | +0.6105 [+0.5915, +0.6296] |
| +10 | 0.0550 | 0.0244 | 18.84 | 0.5262 | 39.4 | 0.6690 | 39.46 | +0.6141 [+0.5928, +0.6350] |
| +15 | 0.0826 | 0.0308 | 25.00 | 0.7490 | 13.6 | 0.6837 | 39.84 | +0.6011 [+0.5782, +0.6241] |
| +20 | 0.0894 | 0.0332 | 27.10 | 0.8247 | 4.9 | 0.6870 | 39.91 | +0.5975 [+0.5744, +0.6205] |

Actual transmitted symbols/image (min/mean/max over all records): 3072/3072.0/3072; AquaSC 3072.


## Budget high: 0.122449 uses/px, 6144 symbols/image


### WEBP q=51 (quality=q, method=4 (PIL defaults otherwise)); capacity 1484 B; test capacity failures 3/432 (0.69%); clean-channel AP50 0.5628, PSNR 40.15, SSIM 0.9601

| SNR | base AP50 | base pooled mAP50 | base PSNR | base SSIM | black % | AquaSC AP50 | AquaSC PSNR | diff AP50 [95% CI] |
|---|---|---|---|---|---|---|---|---|
| -5 | 0.0008 | 0.0000 | 4.52 | 0.0027 | 100.0 | 0.3795 | 32.23 | +0.3787 [+0.3642, +0.3939] |
| +0 | 0.0085 | 0.0114 | 5.01 | 0.0165 | 98.6 | 0.5834 | 36.87 | +0.5749 [+0.5583, +0.5913] |
| +5 | 0.1461 | 0.1326 | 13.99 | 0.2624 | 73.0 | 0.6956 | 39.40 | +0.5495 [+0.5323, +0.5673] |
| +10 | 0.3590 | 0.3246 | 27.54 | 0.6251 | 35.4 | 0.7398 | 40.44 | +0.3808 [+0.3630, +0.3993] |
| +15 | 0.4936 | 0.4423 | 35.86 | 0.8470 | 12.4 | 0.7593 | 40.88 | +0.2657 [+0.2461, +0.2858] |
| +20 | 0.5368 | 0.4804 | 38.61 | 0.9194 | 4.9 | 0.7624 | 40.95 | +0.2257 [+0.2056, +0.2465] |

Actual transmitted symbols/image (min/mean/max over all records): 6144/6144.0/6144; AquaSC 6144.


### JPEG q=24 (quality=q, optimize=True (PIL default 4:2:0). optimize=True chosen on validation only: with PIL defaults the floor is ~1440 B/file, so low and mid budgets are 100% infeasible); capacity 1484 B; test capacity failures 2/432 (0.46%); clean-channel AP50 0.3300, PSNR 36.86, SSIM 0.9398

| SNR | base AP50 | base pooled mAP50 | base PSNR | base SSIM | black % | AquaSC AP50 | AquaSC PSNR | diff AP50 [95% CI] |
|---|---|---|---|---|---|---|---|---|
| -5 | 0.0008 | 0.0000 | 4.52 | 0.0027 | 100.0 | 0.3795 | 32.23 | +0.3787 [+0.3642, +0.3939] |
| +0 | 0.0046 | 0.0099 | 4.83 | 0.0122 | 99.0 | 0.5834 | 36.87 | +0.5788 [+0.5622, +0.5952] |
| +5 | 0.0764 | 0.0621 | 12.19 | 0.2289 | 75.9 | 0.6956 | 39.40 | +0.6192 [+0.6012, +0.6376] |
| +10 | 0.2006 | 0.1583 | 24.51 | 0.5859 | 38.0 | 0.7398 | 40.44 | +0.5392 [+0.5179, +0.5606] |
| +15 | 0.2864 | 0.2231 | 32.58 | 0.8172 | 13.4 | 0.7593 | 40.88 | +0.4729 [+0.4481, +0.4990] |
| +20 | 0.3135 | 0.2433 | 35.33 | 0.8961 | 5.1 | 0.7624 | 40.95 | +0.4489 [+0.4226, +0.4762] |

Actual transmitted symbols/image (min/mean/max over all records): 6144/6144.0/6144; AquaSC 6144.

