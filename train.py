from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Build the neural network
def build_model(input_dim, output_dim):
    model = Sequential([
        Dense(85, input_dim=input_dim, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(18, activation='relu'),  # Hidden layer with 18 neurons
        BatchNormalization(),
        Dropout(0.2),
        Dense(output_dim, activation='softmax')  # Output layer with 9 neurons
    ])
    return model

# Compile the model
model = build_model(input_dim=X_train.shape[1], output_dim=y_train.shape[1])
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32, verbose=2)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")
