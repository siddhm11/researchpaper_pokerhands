
---

# Poker Hand Classification Using XGBoost

This project focuses on classifying poker hands based on their card combinations using a machine learning model, specifically XGBoost. The dataset used is encoded into an 85-bit representation to capture card suits and ranks, and the classifier predicts the class of poker hands.

## 📁 Dataset

### Datasets Used:
1. **Training Data:** `poker-hand-training-true.data`
2. **Testing Data:** `poker-hand-testing.data`

### Dataset Columns:
The dataset contains 11 columns:
- **Card1_Suit, Card1_Rank, ..., Card5_Suit, Card5_Rank:** Suit and rank of the 5 cards.
- **Hand_Class:** Integer representing the type of poker hand.

---

## 🔧 Project Setup

### Prerequisites:
- Python 3.8+
- Libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `xgboost`
  - `matplotlib`

### Installation:
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📜 Code Overview

### Steps:
1. **Data Loading**  
   - The training and testing datasets are loaded using `pandas`.

2. **Feature Encoding**  
   - Card suits and ranks are encoded into an 85-bit representation using one-hot encoding.

3. **Data Splitting**  
   - The training dataset is split into training and validation sets (80/20).

4. **Class Imbalance Handling**  
   - Class weights are calculated dynamically to address imbalances in hand classes.

5. **Model Training**  
   - XGBoost is used with specific parameters to train the model.  
   - Early stopping is employed to prevent overfitting.

6. **Model Evaluation**  
   - The trained model is evaluated on the test dataset using a classification report and confusion matrix.

7. **Feature Importance Visualization**  
   - The top 10 important features are visualized using XGBoost's built-in plotting function.

---

## 🚀 Running the Project

1. **Run the script**  
   Ensure the dataset files (`poker-hand-training-true.data` and `poker-hand-testing.data`) are in the same directory as the script.
   ```bash
   python poker_hand_classification.py
   ```

2. **Output**  
   - **Classification Report:** Metrics like precision, recall, and F1-score for each class.  
   - **Confusion Matrix:** A matrix to evaluate prediction accuracy for each class.  
   - **Feature Importance Plot:** Top 10 most important features influencing the model's predictions.

---

## ⚙️ XGBoost Parameters
| Parameter           | Value           | Description                               |
|---------------------|-----------------|-------------------------------------------|
| `max_depth`         | 8               | Maximum depth of trees.                  |
| `learning_rate`     | 0.1             | Step size shrinkage for weight updates.  |
| `subsample`         | 0.8             | Fraction of samples used for training.   |
| `colsample_bytree`  | 0.8             | Fraction of features used per tree.      |
| `scale_pos_weight`  | Dynamic         | Class imbalance handling.                |
| `objective`         | `multi:softmax` | Multi-class classification objective.    |
| `num_class`         | 10              | Number of classes (0-9).                 |
| `eval_metric`       | `mlogloss`      | Log loss for multi-class classification. |
| `reg_alpha`         | 1               | L1 regularization.                       |
| `reg_lambda`        | 2               | L2 regularization.                       |

---

## 📊 Results
- Classification accuracy is reported with a detailed confusion matrix and a comprehensive classification report.
- Top features influencing the classification are visualized.

---

## 📖 References
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

---

## ✨ Future Work
- Optimize hyperparameters using GridSearchCV.
- Experiment with additional encoding strategies for feature representation.
- Explore alternative machine learning models for improved accuracy.

![image](https://github.com/user-attachments/assets/3034aa01-d76b-4270-9728-3c3051ea64e6)
