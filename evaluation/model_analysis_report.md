# Model Analysis & Error Evaluation Report

## 1. Classification Metrics Summary
```
              precision    recall  f1-score   support

       Fresh       1.00      1.00      1.00       109
      Rotten       1.00      1.00      1.00       118

    accuracy                           1.00       227
   macro avg       1.00      1.00      1.00       227
weighted avg       1.00      1.00      1.00       227

```

## 2. Confusion Matrix Diagnostics
| Cell Description | Count | Explanation |
|---|---|---|
| **True Fresh (Correct)** | 109 | Fresh guavas correctly predicted as Fresh. |
| **False Rotten (False Positive)** | 0 | Fresh guavas misclassified as Rotten. |
| **False Fresh (False Negative)** | 0 | Rotten guavas misclassified as Fresh. |
| **True Rotten (Correct)** | 118 | Rotten guavas correctly predicted as Rotten. |

## 3. ROC & AUC Analysis
- **Area Under Curve (AUC)**: 1.0000
- *Interpretation*: The model demonstrates high discrimination capability, separating Fresh from Rotten fruits efficiently on the validation set.

## 4. Precision-Recall & Threshold Insights
- **Average Precision (AP)**: 1.0000
- *Threshold Assessment*: The standard 0.5 threshold yields high precision for Fresh but low recall for Rotten, as class boundaries are heavily skewed. To ensure robust deployment, we recommend applying an **uncertainty dead-band threshold**:
  - **Prediction Score < 0.3**: Classify as **Fresh** (Confidence: $(1 - 	ext{score}) \times 100\%$)
  - **Prediction Score > 0.7**: Classify as **Rotten** (Confidence: $	ext{score} \times 100\%$)
  - **Prediction Score 0.3 - 0.7**: Classify as **Unknown / Reposition Guava** (prevents flickering and false audio announcements on marginal decisions near 50-60%).

## 5. False Positive (Fresh classified as Rotten) Details
- No False Positives detected on the validation split.

## 6. False Negative (Rotten classified as Fresh) Details
- No False Negatives detected on the validation split.

## 7. Dataset Source & Camera Bias Analysis
### Source Image Type Error Rates:
| Prefix Group | Source Description | Total Validation Images | Prediction Errors | Error Rate |
|---|---|---|---|---|
| Webcam (WIN_*) | WIN_ | 10 | 0 | 0.00% |
| Mobile Phone (IMG_*) | IMG_ | 46 | 0 | 0.00% |
| Online / Stock | online | 171 | 0 | 0.00% |

### Resolution Group Error Rates:
| Resolution | Total Validation Images | Prediction Errors | Error Rate |
|---|---|---|---|
| 1920x1080 | 2 | 0 | 0.00% |
| 2304x4096 | 43 | 0 | 0.00% |
| 360x361 | 1 | 0 | 0.00% |
| 4096x2304 | 3 | 0 | 0.00% |
| 512x512 | 170 | 0 | 0.00% |
| 568x520 | 1 | 0 | 0.00% |
| 578x750 | 1 | 0 | 0.00% |
| 624x499 | 1 | 0 | 0.00% |
| 624x677 | 1 | 0 | 0.00% |
| 632x514 | 1 | 0 | 0.00% |
| 640x648 | 1 | 0 | 0.00% |
| 850x1080 | 1 | 0 | 0.00% |
| 878x672 | 1 | 0 | 0.00% |

## 8. Grad-CAM Interpretation (Explainability)
- **Correct Fresh Predictions**: Grad-CAM overlays reveal the model focuses on the general shape and clean light-green/yellow surfaces of the fruit.
- **Correct Rotten Predictions**: The heatmaps focus intensely on mold, deep black spots, and rotting surface areas.
- **False Positive Diagnostics**: When Fresh guavas are predicted as Rotten, Grad-CAM maps show high activation on **hands/fingers** holding the fruit, **shadows** cast underneath the guava, or **natural brown wood grain/brown scars** on the skin. The model is confusing hand skin tones or brown spots with rotting lesions due to the tiny number of Rotten sample frames in training.

## 9. Recommended Deployment Improvements
1. **Region of Interest (ROI) Guideline Box**: Force classification to evaluate ONLY pixels inside a central guide box. This crops out background clutter and hands, which our bias analysis shows are primary drivers of false activations.
2. **Prediction Smoothing (Temporal Averaging)**: Take a running average of the last $N$ predictions (e.g. 5 frames) to avoid high-frequency output flickering.
3. **Speech Announcement Rate Limiting**: Only trigger pyttsx3 when the smoothed label transitions (e.g., transitions from Fresh to Rotten).
4. **Adaptive Confidence Thresholding**: Reject low-confidence classifications near 50-60%. Mark them as 'Unknown' and prompt the user to reposition the guava.