# ISIC 2024 - Skin Cancer Detection with Machine Learning

**Author:** Eduard Ferré Sánchez  
**Date:** 2025-05-25

---

## Introduction

Skin cancer is among the most prevalent cancers worldwide, where early detection is crucial. The ISIC 2024 challenge focuses on automatically classifying skin lesions as benign or malignant using dermoscopic images alongside patient metadata.

This project leverages **machine learning** and **deep learning** techniques to address challenges such as severe **class imbalance** and **multimodal data integration**, aiming to build a **reliable** and **explainable** diagnostic support system aligned with medical screening practices.

---

## Preprocessing & Data Imbalance

- The ISIC dataset contains **~400,616 benign** and only **343 malignant** cases, representing extreme class imbalance.
- To mitigate this:
  - Applied **SMOTE** on metadata to generate synthetic malignant samples.
  - Extensive **data augmentation** (resizing, flipping, rotations, blurring, perspective distortion) was applied to malignant images.
- Metadata preprocessing included:
  - Filtering irrelevant columns (e.g., `patient_id`, `image_type`)
  - Imputing missing values
  - Encoding categorical variables to preserve data quality and sample size.
- A **custom filter** enhanced lesion borders to improve feature extraction.
- Used **weighted random samplers** in image data loaders to balance class distribution during training and prevent majority class bias.

---

## Best Models & Ensemble Strategies

### Metadata Modeling

- Tested models: **Random Forests**, **Support Vector Machines**, and **Gradient Boosting**.
- Best performance came from an ensemble of:
  - **LightGBM** (efficient gradient boosting)
  - **CatBoost** (native categorical feature handling, reduces overfitting)
  - **XGBoost** (robust regularization)
- This ensemble improved generalization and maintained interpretability compared to image-only models.

### Image Modeling

- Started with baseline CNNs and ResNet variants.
- Selected **ConvNeXt** for its advanced feature extraction, combining ResNet efficiency with Transformer-inspired design.
- Developed a hybrid **ConvNeXt + Vision Transformer (ViT)** model to capture both local and global features.
- These architectures excelled in recognizing subtle malignant signs, especially with strong augmentation.

### Ensemble Strategy for Final Prediction

Evaluated two ensemble methods:

- **Hybrid Worst Case:** Outputs max probability if any model exceeds a threshold (e.g., 0.6), else averages predictions — balancing sensitivity and false positives.
- **Worst Case:** Always selects the highest probability, prioritizing sensitivity.

The Hybrid Worst Case excelled on the public leaderboard; the Worst Case performed better on the private leaderboard.

---

## Explainability Analysis for Medical Use

- Metadata models offer inherent interpretability via feature importance from gradient boosting.
- For image models, applied:
  - **Grad-CAM**
  - Attention visualization on ConvNeXt+ViT
- These methods highlight lesion areas influencing predictions, providing clinicians with intuitive explanations and building trust in the system.

---

## Methodology and Medical Justifications

- Combined **careful preprocessing**, **class imbalance mitigation**, and **advanced multimodal modeling** to improve detection accuracy.
- Integrated metadata and images with robust **ensemble strategies** to enhance reliability.
- Prioritized **sensitivity** to minimize false negatives, reflecting clinical screening priorities where missing malignant cases is critical.
- Ensured **interpretability and trustworthiness** for better acceptance in healthcare.
- Demonstrates potential for **AI-assisted diagnostic tools** to support dermatological screening.

---

## Final Remarks

The hybrid ConvNeXt+ViT model was inspired by the paper:  
**"A novel hybrid ConvNeXt-based approach for enhanced skin lesion classification"**  
[ScienceDirect link](https://www.sciencedirect.com/science/article/abs/pii/S0957417425013430)
