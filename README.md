
# Handwritten Letter Recognizer (A–Z)

This project demonstrates handwritten letter recognition (A–Z) using two different approaches:  
1. **From-Scratch Neural Network (`letter_recognizer.py`)** – implemented entirely with NumPy and core math/linear algebra.  
2. **PyTorch Model (`letter_recognizer_pytorch.py`)** – leveraging the PyTorch deep learning library for a much more concise and powerful implementation.

---

## Purpose
This project was built for educational purposes to:
- Understand the mathematical and linear algebra foundations of neural networks.
- Appreciate how much high-level libraries like PyTorch simplify deep learning.
- Compare training efficiency between custom NumPy implementations and PyTorch.

---

## Results
- **NumPy Neural Net (`letter_recognizer.py`)**
  - Requires ~800 epochs to reach ~84% accuracy.
  - Training loop is simpler but slower and less efficient.
- **PyTorch Neural Net (`letter_recognizer_pytorch.py`)**
  - Reaches ~84% accuracy in only 10 epochs**.
  - Significantly shorter codebase and easier to extend.
  - Epochs take slightly longer to compute due to the more advanced backend, but the efficiency is clear.

This contrast highlights why PyTorch is such a powerful library: fewer lines of code, faster convergence, and easier experimentation.

---

## Installation
Clone the repository and install dependencies:

```bash
git clone <your-repo-url>
cd letter_recognizer
pip install -r requirements.txt
````

---

## Running the Models

Navigate to the `neural_nets/` folder and run either script:

```bash
# NumPy version
python neural_nets/letter_recognizer.py

# PyTorch version
python neural_nets/letter_recognizer_pytorch.py
```

Both scripts will:

* Train a model on the [A\_Z Handwritten Data](https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format) dataset.
* Output predictions and show/save an example handwritten character with its predicted vs. actual label.

---

## Project Structure

```
letter_recognizer/
│
├── data/                     # Dataset (A_Z Handwritten Data.csv)
├── models/                   # Saved models (PyTorch .pth / NumPy .npy weights)
├── neural_nets/
│   ├── letter_recognizer.py          # NumPy neural network
│   └── letter_recognizer_pytorch.py  # PyTorch neural network
│
├── requirements.txt          # Required libraries
└── README.md
```

---

## Disclaimer

This project is intended **solely for educational purposes**.
It is not optimized for production or deployment — the main goal is to learn how neural networks work under the hood and to compare them with modern frameworks like PyTorch.
