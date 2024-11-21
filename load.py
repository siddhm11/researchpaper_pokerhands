import pandas as pd

# Define column names
columns = [
    'Card1_Suit', 'Card1_Rank',
    'Card2_Suit', 'Card2_Rank',
    'Card3_Suit', 'Card3_Rank',
    'Card4_Suit', 'Card4_Rank',
    'Card5_Suit', 'Card5_Rank',
    'Hand_Class'
]

# Load the dataset
train_data = pd.read_csv('poker-hand-training-true.data', header=None, names=columns)
test_data = pd.read_csv('poker-hand-testing.data', header=None, names=columns)

# Display the first few rows of the dataset
print(train_data.head())
print(f"Training Data Shape: {train_data.shape}")
print(f"Testing Data Shape: {test_data.shape}")
