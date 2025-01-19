# Real-Time Cursor Prediction Using LSTM Neural Networks

## Overview
This project involves building a real-time machine learning model that predicts the movement of the computer cursor based on sequential data. The model leverages an LSTM (Long Short-Term Memory) neural network to make predictions about where the cursor will move next. The program runs continuously in the background, learning and improving over time.

## Features
- **Real-Time Prediction**: Predicts the next cursor position in real time based on its movement history.
- **Continuous Learning**: Continuously updates the model with new data to improve prediction accuracy.
- **Dynamic Visualization**: Includes a plotting system to visualize actual vs. predicted cursor movement over time, with interactive features like a time slider.
- **Automated Startup**: Configured to run automatically on system startup and wake from sleep.
- **Periodic Model Saving**: Automatically saves the model every minute to prevent loss of progress.

## Technologies Used
- **Python**: Core programming language.
- **PyTorch**: Used for building and training the LSTM model.
- **PyAutoGUI**: For capturing cursor movement.
- **Matplotlib**: For visualizing actual and predicted cursor positions.
- **Batch Scripting**: For automating program execution on startup.

## How It Works
1. **Data Collection**: The program continuously tracks the cursor's `(x, y)` position at regular intervals. Velocity and acceleration features are computed for better predictions.
2. **Model Training**: The LSTM neural network trains on collected data to predict the next cursor position.
3. **Real-Time Execution**: The model predicts and visualizes the cursor's movement while running in the background.
4. **Periodic Saving**: The model state is saved every minute to a `.pth` file for future use.
5. **Startup Automation**: A batch script ensures the program runs automatically when the computer boots or wakes from sleep.

## Installation
### Prerequisites
- Python 3.11 or higher
- Virtual Environment (`venv`)

### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/cursor-predictor.git
   cd cursor-predictor
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv anotherenv
   call anotherenv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the script:
   ```bash
   python cursor_predictor.py
   ```

## Configuration
### Batch File for Startup
To run the program on startup:
1. Create a batch file (e.g., `start_cursor_predictor.bat`) with the following content:
   ```bat
   @echo off
   cd "C:\path\to\project"
   call "C:\path\to\project\anotherenv\Scripts\activate"
   "C:\path\to\project\anotherenv\Scripts\python.exe" "C:\path\to\project\cursor_predictor.py"
   ```
2. Place the batch file in the Windows Startup folder:
   ```
   %APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup
   ```

## Usage
1. Start the program manually or ensure it runs on system startup.
2. Observe the predictions and performance in the terminal.
3. Use the visualization plot to analyze actual vs. predicted cursor movements interactively.

## Project Structure
```
.
├── cursor_predictor.py       # Main script for real-time prediction
├── requirements.txt          # List of dependencies
├── startup_debug.txt         # Log file for debugging startup issues
└── anotherenv/               # Virtual environment directory
```

## Future Improvements
- Incorporate more advanced features like multi-user datasets.
- Implement faster or alternative prediction models (e.g., Transformers).
- Enhance visualization with more interactive tools.
