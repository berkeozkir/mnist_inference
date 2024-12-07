# MNIST Digit Classifier: Training and Inference

This project is a web application built with **Streamlit** that allows users to interact with a convolutional neural network (CNN) trained on the MNIST dataset. Users can draw a digit on a canvas, and the application predicts the digit in real time.

---

## Features

- **Interactive Canvas**: Draw digits directly on the canvas using your mouse or touch input.
- **Fast Predictions**: Classify handwritten digits using a PyTorch-based CNN.
- **Visualization**: View the drawn digit and its processed image before classification.
- **User-Friendly Interface**: Designed with Streamlit for seamless interaction.

---

## Technology Stack

- **Frontend**: Streamlit with `streamlit-drawable-canvas`
- **Backend**: PyTorch for the CNN model
- **Visualization**: Matplotlib and Plotly for graphical output

---

## How to Run the Application

### Prerequisites

Ensure you have the following installed:
- Python 3.8 or later
- Pip (Python package manager)

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/mnist-digit-classifier.git
   cd mnist-digit-classifier

2. **Install the required Python packages**:
   ```bash
   pip install -r requirements.txt

3. **Run the Streamlit application**:
   ```bash
   streamlit run mnistUI.py

3. **Access the application**:
   Open your browser and navigate to http://localhost:8501.
