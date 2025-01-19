import time
import pyautogui
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

# --------------------------
# 1. Model Definition
# --------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)  # Predict (x, y)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])  # Last time step output
        return out, hidden

    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))

# --------------------------
# 2. Hyperparameters
# --------------------------
SEQ_LENGTH = 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 1
TRAINING_STEPS = 5
SLEEP_INTERVAL = 0.05

# --------------------------
# 3. Initialize Model and Training Tools
# --------------------------
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
hidden_state = model.init_hidden(BATCH_SIZE)

# Buffers for recent data
x_buffer = deque(maxlen=SEQ_LENGTH)
y_buffer = deque(maxlen=SEQ_LENGTH)
velocity_buffer = deque(maxlen=SEQ_LENGTH - 1)

def calculate_velocity(x1, y1, x2, y2, dt):
    if dt == 0:
        return 0, 0
    return (x2 - x1) / dt, (y2 - y1) / dt

# Metrics storage
actual_positions = []
predicted_positions = []
timestamps = []

start_time = time.time()

# --------------------------
# 5. Continuous Loop
# --------------------------
print("Starting continuous capture and learning. Press Ctrl+C to stop.")
try:
    prev_x, prev_y = None, None
    prev_time = None

    while True:
        # Capture current cursor position
        current_x, current_y = pyautogui.position()
        current_time = time.time()
        elapsed_time = current_time - start_time

        # Calculate velocity if previous position exists
        if prev_x is not None and prev_time is not None:
            vx, vy = calculate_velocity(prev_x, prev_y, current_x, current_y, current_time - prev_time)
            velocity_buffer.append((vx, vy))

        # Update previous position and time
        prev_x, prev_y = current_x, current_y
        prev_time = current_time

        # Append to buffers
        x_buffer.append(current_x)
        y_buffer.append(current_y)

        # Store actual positions and time
        actual_positions.append((current_x, current_y))

        # If buffer is full, predict and train
        if len(x_buffer) == SEQ_LENGTH and len(velocity_buffer) == SEQ_LENGTH - 1:
            # Prepare input for the model (x, y, vx, vy)
            seq_input = torch.tensor([[ [x, y, vx, vy] for (x, y), (vx, vy) in zip(zip(x_buffer, y_buffer), velocity_buffer) ]],
                                     dtype=torch.float32)

            # Predict next position
            model.eval()
            with torch.no_grad():
                pred, hidden_state = model(seq_input, hidden_state)
            predicted_x, predicted_y = pred[0].tolist()

            # Store predicted positions and timestamp
            predicted_positions.append((predicted_x, predicted_y))
            timestamps.append(elapsed_time)  # Only log timestamp when a prediction is made

            print(f"Actual: ({current_x}, {current_y}), Predicted: ({predicted_x:.2f}, {predicted_y:.2f})")

            # Train model
            model.train()
            for _ in range(TRAINING_STEPS):
                optimizer.zero_grad()
                label = torch.tensor([ [current_x, current_y] ], dtype=torch.float32)
                output, _ = model(seq_input, model.init_hidden(BATCH_SIZE))
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

        # Sleep for interval
        time.sleep(SLEEP_INTERVAL)

except KeyboardInterrupt:
    pass

# --------------------------
# 6. Plot the Results
# --------------------------
print("Plotting results...")

# Ensure all arrays are the same length
min_length = min(len(actual_positions), len(predicted_positions), len(timestamps))
actual_positions = actual_positions[:min_length]
predicted_positions = predicted_positions[:min_length]
timestamps = timestamps[:min_length]

# Convert data to numpy arrays for easier plotting
actual_positions = np.array(actual_positions)
predicted_positions = np.array(predicted_positions)
timestamps = np.array(timestamps)

# Create a color gradient based on time
colors = plt.cm.coolwarm((timestamps - timestamps.min()) / (timestamps.max() - timestamps.min()))

# Plot actual and predicted positions with a slider
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
actual_line, = ax.plot(actual_positions[:, 0], actual_positions[:, 1], label="Actual", alpha=0.7, color='blue', linestyle='-', linewidth=1)
predicted_line, = ax.plot(predicted_positions[:, 0], predicted_positions[:, 1], label="Predicted", alpha=0.7, color='orange', linestyle='--', linewidth=1)

# Add highlight points
highlight_actual, = ax.plot([], [], 'o', color='red', label='Highlighted Actual')
highlight_predicted, = ax.plot([], [], 'o', color='green', label='Highlighted Predicted')

# Add labels and legend
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Cursor Movement Prediction")
plt.legend()

# Add a slider
ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
time_slider = Slider(ax_slider, 'Time', timestamps.min(), timestamps.max(), valinit=timestamps.min(), valstep=(timestamps.max() - timestamps.min()) / len(timestamps))

def update(val):
    time_idx = (np.abs(timestamps - time_slider.val)).argmin()
    highlight_actual.set_data([actual_positions[time_idx, 0]], [actual_positions[time_idx, 1]])
    highlight_predicted.set_data([predicted_positions[time_idx, 0]], [predicted_positions[time_idx, 1]])
    fig.canvas.draw_idle()

time_slider.on_changed(update)

plt.show()
