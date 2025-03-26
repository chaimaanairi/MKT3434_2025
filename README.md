# MKT3434 - Homework Assignment #1

## GUI Enhancements for Data Handling and Model Completion

### Dr. Ertuğrul BAYRAKTAR  
### Chaimaa Nairi - 23501057  

---

## 🚀 Overview
This project enhances the base GUI framework provided for the MKT3434 course. The modifications include improvements in data handling, model training, and visualization. Key additions include:

- **Support for Loss Functions:**
  - Mean Squared Error (MSE), Mean Absolute Error (MAE), and Huber Loss for regression models.
  - Cross-Entropy and Hinge Loss for classification models.
- **Enhanced Support Vector Machine (SVM) with:**
  - Kernel selection: `linear`, `rbf`, `polynomial`
  - Additional hyperparameters: `C` (Regularization) and `epsilon` (SVR only)
- **Gaussian Naïve Bayes (GaussianNB):**
  - Configurable `var_smoothing` parameter
  - User-defined prior probabilities
- **Missing Data Handling Options:**
  - Mean Imputation
  - Interpolation
  - Forward/Backward Fill
- **SVR Testing on the Boston Housing Dataset** with performance evaluation.
- **Updated GUI** to include new features and selections.

---

## 🛠 New Features & Enhancements

### 1️⃣ Loss Function Selection for Model Training
| Loss Type      | Available Functions       |
|---------------|--------------------------|
| Regression    | MSE, MAE, Huber Loss      |
| Classification | Cross-Entropy, Hinge Loss |

- **GUI Update:** Added dropdown menu for selecting loss functions.

### 2️⃣ Improved SVM Model Support
- **Classification (SVC) & Regression (SVR)**
- **User-selectable Kernels:** `linear`, `rbf`, `polynomial`
- **Hyperparameter Tuning:** `C`, `epsilon`
- **GUI Update:** Parameter selection menu added.

### 3️⃣ Gaussian Naïve Bayes (GaussianNB) Implementation
- **User-configurable `var_smoothing` parameter**
- **Customizable Prior Probabilities** (Uniform or User-Defined)
- **GUI Update:** Added input fields for priors and smoothing.

### 4️⃣ Missing Data Handling Methods
#### Options Available:
| Handling Method      | Description                            |
|---------------------|----------------------------------|
| Mean Imputation    | Replaces missing values with the mean of the column. |
| Interpolation      | Estimates missing values using neighboring values. |
| Forward/Backward Fill | Fills missing values using previous/next available values. |

- **Implementation:** Uses `SimpleImputer` from `scikit-learn`.
- **GUI Update:** Dropdown menu for missing data handling.

### 5️⃣ SVR Testing on the Boston Housing Dataset
- **Dataset:** `datasets.load_boston()`
- **Performance Metrics:** MSE, MAE
- **Visualization:** Results displayed in the GUI.

---

