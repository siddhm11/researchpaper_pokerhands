import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt

# Load dataset and define column names
columns = [
    'Card1_Suit', 'Card1_Rank',
    'Card2_Suit', 'Card2_Rank',
    'Card3_Suit', 'Card3_Rank',
    'Card4_Suit', 'Card4_Rank',
    'Card5_Suit', 'Card5_Rank',
    'Hand_Class'
]

train_data = pd.read_csv('poker-hand-training-true.data', header=None, names=columns)
test_data = pd.read_csv('poker-hand-testing.data', header=None, names=columns)

# Encode input features into 85-bit representation
def encode_to_85_bits(data):
    suits = ['Card1_Suit', 'Card2_Suit', 'Card3_Suit', 'Card4_Suit', 'Card5_Suit']
    ranks = ['Card1_Rank', 'Card2_Rank', 'Card3_Rank', 'Card4_Rank', 'Card5_Rank']

    suit_encoded = pd.get_dummies(data[suits], prefix='suit', drop_first=False)
    rank_encoded = pd.get_dummies(data[ranks], prefix='rank', drop_first=False)

    return pd.concat([suit_encoded, rank_encoded], axis=1).values

# Encode features and labels
X_train = encode_to_85_bits(train_data)
X_test = encode_to_85_bits(test_data)

y_train = train_data['Hand_Class'].values
y_test = test_data['Hand_Class'].values

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Create the DMatrix for early stopping
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

# Compute scale_pos_weight dynamically for class imbalance
class_weights = {i: len(y_train) / (len(np.unique(y_train)) * np.bincount(y_train)[i]) for i in np.unique(y_train)}

# Define XGBoost parameters
params = {
    'max_depth': 8,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': class_weights,
    'objective': 'multi:softmax',
    'num_class': 10,
    'random_state': 42,
    'verbosity': 1,
    'eval_metric': 'mlogloss',
    'reg_alpha': 1,  # L1 regularization
    'reg_lambda': 2  # L2 regularization
}

# Train the model with early stopping
evals = [(dtrain, 'train'), (dval, 'validation')]
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    evals=evals,
    early_stopping_rounds=10,
    verbose_eval=True
)

# Predict on the test set
y_pred = bst.predict(dtest)

# Classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Feature importance visualization
xgb.plot_importance(bst, max_num_features=10)
plt.title("Top 10 Important Features")
plt.show()
