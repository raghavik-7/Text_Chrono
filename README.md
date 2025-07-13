# TextChrono

A machine learning project for text classification using recurrent neural networks. This project implements vanilla RNN and LSTM models from scratch for sentiment analysis on movie reviews.

## Overview

TextChrono is a deep learning project that demonstrates the implementation of recurrent neural networks for natural language processing tasks. The project includes:

- Custom implementations of Vanilla RNN and LSTM cells using NumPy
- Sentiment analysis on IMDB movie reviews dataset
- Interactive prediction interface for user input
- Support for both training and inference modes

## Project Structure

```
TextChrono/
├── main.py                 # Main training and inference script
├── requirements.txt        # Python dependencies
├── data/                   # Dataset directory
│   ├── book_quotes.txt     # Book quotes dataset
│   └── reviews.csv         # IMDB reviews dataset
├── models/                 # Neural network implementations
│   ├── rnn_cell.py         # Vanilla RNN cell implementation
│   ├── lstm_cell.py        # LSTM cell implementation
│   ├── stacked.py          # Stacked RNN/LSTM architectures
│   └── bptt.py            # Backpropagation through time utilities
├── utils/                  # Utility functions
│   └── data_loader.py      # Data loading and preprocessing
├── model_weights/          # Saved model weights
│   ├── rnn_weights.npz     # RNN model weights
│   ├── rnn_clf.npz         # RNN classifier weights
│   ├── lstm_weights.npz    # LSTM model weights
│   └── lstm_clf.npz        # LSTM classifier weights
└── notebooks/              # Jupyter notebooks for experimentation
```

## Features

### Model Implementations
- **Vanilla RNN**: Basic recurrent neural network with tanh activation
- **LSTM**: Long Short-Term Memory network with forget, input, and output gates
- **Stacked Architectures**: Support for multi-layer RNN/LSTM networks

### Training Features
- Automatic data downloading from IMDB dataset
- Vocabulary building with configurable minimum frequency and size
- Batch processing with configurable batch sizes
- Early stopping based on validation accuracy
- Gradient clipping for training stability
- Dropout regularization during training
- Learning rate warmup and scheduling

### Inference Features
- Interactive sentiment prediction interface
- Support for loading pre-trained models
- Real-time text processing and classification

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd TextChrono
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training a Model

Train an LSTM model:
```bash
python main.py --model lstm --save
```

Train an RNN model:
```bash
python main.py --model rnn --save
```

### Loading Pre-trained Models

Load and use a pre-trained model:
```bash
python main.py --model lstm --load
```

### Interactive Prediction

After training or loading a model, the script provides an interactive interface for sentiment prediction:

```
Enter a review (or 'quit' to exit): This movie was absolutely fantastic!
Predicted sentiment: positive (confidence: 0.85)
```

## Configuration

Key parameters can be modified in `main.py`:

- `MAX_LEN`: Maximum sequence length (default: 50)
- `BATCH_SIZE`: Training batch size (default: 8)
- `HIDDEN_SIZE`: Hidden layer size (default: 256)
- `EPOCHS`: Number of training epochs (default: 200)
- `DROPOUT_PROB`: Dropout probability (default: 0.1)
- `EMBEDDING_DIM`: Word embedding dimension (default: 256)

## Data

The project uses the IMDB movie reviews dataset for sentiment analysis. The dataset is automatically downloaded on first run and contains:

- Movie reviews with positive/negative labels
- Preprocessed text with HTML tags and punctuation removed
- Tokenized and encoded sequences for model training

## Model Architecture

### Vanilla RNN
- Input embedding layer
- Single RNN cell with tanh activation
- Dropout layer for regularization
- Linear classifier head

### LSTM
- Input embedding layer
- LSTM cell with forget, input, candidate, and output gates
- Dropout layer for regularization
- Linear classifier head

Both models use backpropagation through time (BPTT) for training and include gradient clipping to prevent exploding gradients.

## Performance

The models demonstrate:
- Custom implementation of RNN/LSTM cells
- Proper gradient flow and weight updates
- Training stability with gradient clipping
- Interactive inference capabilities

## Dependencies

- numpy: Numerical computations
- matplotlib: Plotting and visualization
- requests: Data downloading
- jupyter: Notebook environment
- pandas: Data manipulation

## Contributing

Contributions are welcome. Please ensure code follows the existing style and includes appropriate documentation. 
