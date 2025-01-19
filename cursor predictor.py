import time
import pyautogui
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# --------------------------
# 1. Model Definition
# --------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=32, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)  # Predict (x, y)

    def forward(self, x, hidden):
        # x shape: (batch, seq_len, input_size=2)
        out, hidden = self.lstm(x, hidden)
        # Use last time step's output for prediction
        out = self.fc(out[:, -1, :])
        return out, hidden

    def init_hidden(self, batch_size=1):
        # Hidden and cell state for LSTM
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))

# --------------------------
# 2. Hyperparameters
# --------------------------
SEQ_LENGTH = 10       # Number of timesteps used as input
LEARNING_RATE = 1e-3
BATCH_SIZE = 1        # We'll train on one sequence at a time in this toy example
TRAINING_STEPS = 5    # Number of mini-steps to train on new data
SLEEP_INTERVAL = 0.1  # Seconds between cursor checks

# --------------------------
# 3. Initialize model, loss, optimizer
# --------------------------
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
hidden_state = model.init_hidden(BATCH_SIZE)

# --------------------------
# 4. Buffers to store recent data
# --------------------------
x_buffer = deque(maxlen=SEQ_LENGTH)
y_buffer = deque(maxlen=SEQ_LENGTH)

# --------------------------
# 5. Continuous Loop
# --------------------------
print("Starting continuous capture and learning. Press Ctrl+C to stop.")
try:
    while True:
        # 5.1 Capture current cursor position
        current_x, current_y = pyautogui.position()
        
        # Normalize or scale if needed. For simplicity, we use raw coordinates here.
        x_buffer.append(current_x)
        y_buffer.append(current_y)
        
        # 5.2 Once we have enough points in the buffer, train and predict
        if len(x_buffer) == SEQ_LENGTH:
            # Prepare input for the model (shape: [batch_size, seq_len, input_size])
            seq_input = torch.tensor([[ [x, y] for x, y in zip(x_buffer, y_buffer) ]],
                                     dtype=torch.float32)
            
            # Predict next position before training (for demonstration)
            model.eval()
            with torch.no_grad():
                pred, hidden_state = model(seq_input, hidden_state)
            
            # Print prediction
            predicted_x, predicted_y = pred[0].tolist()
            print(f"Predicted Next Position: ({predicted_x:.2f}, {predicted_y:.2f})")
            
            # Simulate target as the next actual coordinate you might receive 
            # once time moves forward. Here we'll re-fetch cursor or you could
            # wait for the next time step. This is just a simplified example
            # for continuous training on the current target. 
            #
            # In a real-time setting, you'd do partial fitting as new data arrives
            # and treat the *future* position as the label for the current sequence.
            # For demonstration, let's train on the same last point to show the loop.
            
            model.train()
            for _ in range(TRAINING_STEPS):
                # Forward + Backward + Optimize
                optimizer.zero_grad()
                
                # Use the final point in the sequence as the "label" 
                # in this simplistic example
                label = torch.tensor([ [current_x, current_y] ], dtype=torch.float32)
                
                output, hidden_state = model(seq_input, model.init_hidden(BATCH_SIZE))
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
            
        # Sleep briefly before checking again
        time.sleep(SLEEP_INTERVAL)

except KeyboardInterrupt:
    pass
