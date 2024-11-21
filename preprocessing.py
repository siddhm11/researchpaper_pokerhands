import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.metrics import confusion_matrix, classification_report
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

print(f"Training Data Shape: {train_data.shape}")
print(f"Testing Data Shape: {test_data.shape}")

# Encode features into 85-bit representation
def encode_to_85_bits(data):
    suits = ['Card1_Suit', 'Card2_Suit', 'Card3_Suit', 'Card4_Suit', 'Card5_Suit']
    ranks = ['Card1_Rank', 'Card2_Rank', 'Card3_Rank', 'Card4_Rank', 'Card5_Rank']

    suit_encoded = pd.get_dummies(data[suits], prefix='suit', drop_first=False)
    rank_encoded = pd.get_dummies(data[ranks], prefix='rank', drop_first=False)

    return pd.concat([suit_encoded, rank_encoded], axis=1).values

X_train = encode_to_85_bits(train_data)
X_test = encode_to_85_bits(test_data)

# Encode labels into one-hot format
encoder = OneHotEncoder(sparse_output=False)
y_train = encoder.fit_transform(train_data[['Hand_Class']])
y_test = encoder.transform(test_data[['Hand_Class']])

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Compute class weights to handle imbalance
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data['Hand_Class']),
    y=train_data['Hand_Class']
)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

print(f"Encoded Training Features Shape: {X_train.shape}")
print(f"Encoded Training Labels Shape: {y_train.shape}")

# Build the neural network
def build_model(input_dim, output_dim):
    model = Sequential([
        Dense(256, input_dim=input_dim, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(output_dim, activation='softmax')  # Output layer with 10 classes
    ])
    return model

# Define learning rate schedule
lr_schedule = ExponentialDecay(
    initial_learning_rate=0.0005,
    decay_steps=10000,
    decay_rate=0.9
)

# Compile the model
model = build_model(input_dim=X_train.shape[1], output_dim=y_train.shape[1])
optimizer = Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,  # Increased number of epochs for better convergence
    batch_size=32,
    class_weight=class_weights_dict,
    verbose=2
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Generate classification report
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# Plot accuracy and loss graphs
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.tight_layout()
plt.show()
