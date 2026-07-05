# Guava Dataset Quality Check Report

## 1. Class Imbalance Report
- **Fresh**: 533 images (47.04%)
- **Rotten**: 600 images (52.96%)
- **Total Images**: 1133

## 2. Corrupted & Unreadable Images
- No corrupted or unreadable images detected.

## 3. Duplicate Images
- No duplicate images detected based on file hash comparisons.

## 4. Blurry Images (Laplacian Variance < 100)
Total blurry images detected: 234 / 1133

| Image Name | Class | Laplacian Variance |
|---|---|---|
| IMG_20260702_150859331_HDR.jpg | Fresh | 6.07 |
| IMG_20260702_150842631_HDR.jpg | Fresh | 6.17 |
| WIN_20260702_16_03_58_Pro.jpg | Fresh | 6.35 |
| IMG_20260702_150928931_HDR.jpg | Fresh | 6.84 |
| IMG_20260629_160220197_HDR.jpg | Fresh | 7.66 |
| IMG_20260702_152124256_HDR.jpg | Fresh | 7.75 |
| IMG_20260702_154208445_HDR.jpg | Rotten | 8.06 |
| WIN_20260629_16_28_39_Pro.jpg | Rotten | 8.26 |
| IMG_20260702_154244642_HDR.jpg | Rotten | 8.72 |
| IMG_20260702_152230465_HDR.jpg | Fresh | 9.10 |
| IMG_20260702_154234922_HDR.jpg | Rotten | 9.20 |
| IMG_20260702_154212310_HDR.jpg | Rotten | 9.20 |
| IMG_20260702_154238176_HDR.jpg | Rotten | 9.43 |
| IMG_20260629_160215057_HDR.jpg | Fresh | 11.14 |
| WIN_20260629_16_35_51_Pro.jpg | Fresh | 11.21 |
| IMG_20260702_150334436_HDR.jpg | Fresh | 11.66 |
| WIN_20260629_16_24_35_Pro.jpg | Fresh | 11.79 |
| IMG_20260702_152117870_HDR.jpg | Fresh | 11.88 |
| IMG_20260702_154222511_HDR.jpg | Rotten | 12.06 |
| WIN_20260629_16_22_30_Pro.jpg | Fresh | 12.14 |

*...and 214 more blurry images.*

## 5. Image Resolutions Distribution
- **512x512**: 861 images
- **2304x4096**: 196 images
- **1920x1080**: 18 images
- **4096x2304**: 13 images
- **332x360**: 1 images
- **645x360**: 1 images
- **462x250**: 1 images
- **800x533**: 1 images
- **2272x2244**: 1 images
- **1638x2913**: 1 images
- **1975x3512**: 1 images
- **1765x3139**: 1 images
- **1739x3092**: 1 images
- **1812x3222**: 1 images
- **1911x3398**: 1 images
- **1916x3407**: 1 images
- **1959x3479**: 1 images
- **525x350**: 1 images
- **624x677**: 1 images
- **734x629**: 1 images
- **734x766**: 1 images
- **578x750**: 1 images
- **878x672**: 1 images
- **657x658**: 1 images
- **1007x870**: 1 images
- **372x311**: 1 images
- **640x648**: 1 images
- **803x1080**: 1 images
- **734x1080**: 1 images
- **1080x944**: 1 images
- **1028x716**: 1 images
- **843x783**: 1 images
- **632x514**: 1 images
- **670x675**: 1 images
- **1034x1080**: 1 images
- **628x532**: 1 images
- **822x734**: 1 images
- **624x499**: 1 images
- **632x605**: 1 images
- **762x791**: 1 images
- **568x520**: 1 images
- **517x499**: 1 images
- **853x771**: 1 images
- **507x900**: 1 images
- **800x682**: 1 images
- **360x361**: 1 images
- **200x184**: 1 images
- **1063x1080**: 1 images
- **850x1080**: 1 images
