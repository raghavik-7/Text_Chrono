import os
import argparse
import random
from utils.data_loader import load_imdb, build_vocab, batch_generator, tokenize
from models.rnn_cell import VanillaRNNCell
from models.lstm_cell import LSTMCell
from models.bptt import train_rnn, train_lstm
from models.stacked import StackedRNN, StackedLSTM
import numpy as np
from collections import Counter

DATA_PATH = 'TextChrono/data/reviews.csv'
MAX_LEN = 50
BATCH_SIZE = 8
HIDDEN_SIZE = 256
EPOCHS = 200
DROPOUT_PROB = 0.1  # Lowered from 0.5do
WEIGHTS_DIR = 'TextChrono/model_weights'
EMBEDDING_DIM = 256  # Reduced for memory safety
OVERFIT_SANITY_CHECK = False  # Set to False to train on the full dataset

if not os.path.exists(WEIGHTS_DIR):
    os.makedirs(WEIGHTS_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['rnn', 'lstm'], default='lstm')
    parser.add_argument('--save', action='store_true', help='Save trained weights after training')
    parser.add_argument('--load', action='store_true', help='Load weights for inference')
    args = parser.parse_args()
    args.num_layers = 1
    print("Loading IMDB data...")
    texts, labels = load_imdb(DATA_PATH, max_samples=2000)
    print(f"Loaded {len(texts)} samples.")
    print("Building vocabulary...")
    vocab = build_vocab(texts, min_freq=2, max_size=5000)
    print(f"Vocab size: {len(vocab)}")
    # Print class distribution
    label_counts = Counter(labels)
    print(f"Class distribution: {label_counts}")
    total_labels = sum(label_counts.values())
    for cls, count in label_counts.items():
        frac = count / total_labels
        if frac < 0.4:
            print(f"[WARNING] Class {cls} is underrepresented: {frac*100:.1f}% of data.")
    # Shuffle and split data
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    split = int(0.8 * len(combined))
    train_data = combined[:split]
    val_data = combined[split:]
    train_texts, train_labels = zip(*train_data)
    val_texts, val_labels = zip(*val_data)
    print(f"Training samples: {len(train_texts)}, Validation samples: {len(val_texts)}")
    print("Creating batches...")
    train_batches = list(batch_generator(train_texts, train_labels, vocab, batch_size=BATCH_SIZE, max_len=MAX_LEN))
    val_batches = list(batch_generator(val_texts, val_labels, vocab, batch_size=BATCH_SIZE, max_len=MAX_LEN, shuffle=False))
    if OVERFIT_SANITY_CHECK:
        print("[SANITY CHECK] Overfitting a single small batch (8 samples). This should reach near 100% accuracy if the model is correct.")
        train_batches = train_batches[:1]
        val_batches = val_batches[:1]
    input_size = len(vocab)
    output_size = 2
    # Embedding layer
    embedding = np.random.randn(input_size, EMBEDDING_DIM).astype(np.float32) / np.sqrt(input_size)
    d_embedding = np.zeros((input_size, EMBEDDING_DIM), dtype=np.float32)
    rnn_input_size = EMBEDDING_DIM  # Use embedding dim as input to RNN/LSTM
    # Classifier head
    W_out = np.random.randn(output_size, HIDDEN_SIZE).astype(np.float32) * 0.01
    b_out = np.zeros((output_size, 1), dtype=np.float32)
    # Model
    if args.model == 'rnn':
        print("Initializing Vanilla RNN...")
        model = VanillaRNNCell(rnn_input_size, HIDDEN_SIZE)
    else:
        print("Initializing LSTM...")
        model = LSTMCell(rnn_input_size, HIDDEN_SIZE)
    print(f"Using model: {args.model.upper()}")
    # Load weights if requested
    if args.load:
        print("Loading model and classifier weights...")
        model_weights_path = os.path.join(WEIGHTS_DIR, f'{args.model}_weights.npz')
        clf_weights_path = os.path.join(WEIGHTS_DIR, f'{args.model}_clf.npz')
        if os.path.exists(model_weights_path):
            weights = np.load(model_weights_path, allow_pickle=True)
            for k in weights:
                if hasattr(model, k):
                    setattr(model, k, weights[k])
        if os.path.exists(clf_weights_path):
            clf = np.load(clf_weights_path)
            W_out = clf['W_out']
            b_out = clf['b_out']
    else:
        print(f"Training {args.model.upper()} and classifier head...")
        lr = 0.01  # Lower learning rate for stability
        # Print first batch for debugging
        first_batch_printed = False
        best_val_loss = float('inf')
        epochs_no_improve = 0
        early_stop_patience = 50  # Allow more epochs for overfitting
        for epoch in range(EPOCHS):
            # Learning rate warmup
            lr = 0.005 if epoch < 3 else 0.01
            epoch_loss = 0
            correct, total = 0, 0
            for batch_x, batch_y in train_batches:
                seq_len = batch_x.shape[1]
                batch_size = batch_x.shape[0]
                # Use embedding lookup instead of one-hot
                emb = embedding[batch_x]  # (batch_size, seq_len, embedding_dim)
                x_seq = np.transpose(emb, (1, 2, 0))  # (seq_len, embedding_dim, batch_size)
                train_mode = True
                if isinstance(model, (VanillaRNNCell, StackedRNN)):
                    hs = model.forward(x_seq)
                    h_last = hs[-1] if isinstance(hs, list) else hs[-1]
                    h0_shape = np.asarray(hs[0]).shape if isinstance(hs, list) else np.asarray(hs).shape
                else:
                    hs, cs = model.forward(x_seq)
                    h_last = hs[-1]
                    h0_shape = np.asarray(hs[0]).shape if isinstance(hs, list) else np.asarray(hs).shape
                # Dropout during training only
                if train_mode:
                    dropout_mask = (np.random.rand(*h_last.shape) > DROPOUT_PROB) / (1.0 - DROPOUT_PROB)
                else:
                    dropout_mask = np.ones_like(h_last)
                h_last_dropout = h_last * dropout_mask
                logits = np.dot(W_out, h_last_dropout) + b_out
                logits -= np.max(logits, axis=0, keepdims=True)  # Softmax stability
                exp_logits = np.exp(logits)
                probs = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)
                loss = -np.sum(np.log(probs[batch_y, np.arange(batch_size)] + 1e-9)) / batch_size
                epoch_loss += loss
                preds = np.argmax(probs, axis=0)
                correct += np.sum(preds == batch_y)
                total += batch_size
                # Backprop for classifier head
                dlogits = probs
                dlogits[batch_y, np.arange(batch_size)] -= 1
                dlogits /= batch_size
                dW_out = np.dot(dlogits, h_last_dropout.T)
                db_out = np.sum(dlogits, axis=1, keepdims=True)
                print(f"Epoch {epoch+1} grad_norm: {np.linalg.norm(dW_out):.4f}")
                dh = np.dot(W_out.T, dlogits) * dropout_mask
                dhs = [np.zeros(h0_shape) for _ in range(seq_len-1)] + [dh]
                if isinstance(model, VanillaRNNCell):
                    dxs = model.backward(dhs, clip=5.0)
                    model.step(lr)
                else:
                    if isinstance(model, StackedRNN):
                        dxs = model.backward(dhs, clip=5.0)
                        model.step(lr)
                    else:
                        dcs = [np.zeros_like(cs[0]) for _ in range(seq_len)]
                        dxs = model.backward(dhs, dcs, clip=5.0)
                        model.step(lr)
                W_out[...] -= lr * dW_out
                b_out[...] -= lr * db_out
                # Embedding update: only update each token once per batch
                updated_indices = set()
                for t in range(seq_len):
                    for b in range(batch_size):
                        idx = batch_x[b, t]
                        if idx not in updated_indices:
                            embedding[idx] -= lr * d_embedding[idx]
                            updated_indices.add(idx)
                d_embedding.fill(0)
            avg_loss = epoch_loss / (total // BATCH_SIZE)
            acc = correct / total
            if acc > 0.9:
                print(f"High training accuracy detected: {acc:.4f}. Stopping early.")
                break
            # Validation
            val_loss = 0
            val_correct, val_total = 0, 0
            for batch_x, batch_y in val_batches:
                seq_len = batch_x.shape[1]
                batch_size = batch_x.shape[0]
                emb = embedding[batch_x]
                x_seq = np.transpose(emb, (1, 2, 0))
                train_mode = False
                if isinstance(model, (VanillaRNNCell, StackedRNN)):
                    hs = model.forward(x_seq)
                    h_last = hs[-1] if isinstance(hs, list) else hs[-1]
                    h0_shape = np.asarray(hs[0]).shape if isinstance(hs, list) else np.asarray(hs).shape
                else:
                    hs, cs = model.forward(x_seq)
                    h_last = hs[-1]
                    h0_shape = np.asarray(hs[0]).shape if isinstance(hs, list) else np.asarray(hs).shape
                dropout_mask = np.ones_like(h_last)
                h_last_dropout = h_last * dropout_mask
                logits = np.dot(W_out, h_last_dropout) + b_out
                logits -= np.max(logits, axis=0, keepdims=True)
                exp_logits = np.exp(logits)
                probs = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)
                loss = -np.sum(np.log(probs[batch_y, np.arange(batch_size)] + 1e-9)) / batch_size
                val_loss += loss
                preds = np.argmax(probs, axis=0)
                val_correct += np.sum(preds == batch_y)
                val_total += batch_size
            avg_val_loss = val_loss / (val_total // BATCH_SIZE)
            val_acc = val_correct / val_total
            print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, acc={acc:.4f} | val_loss={avg_val_loss:.4f}, val_acc={val_acc:.4f}")
            # Save best model
            if val_acc > best_val_loss:
                print("[INFO] Saving best model (val_acc improved)")
                model_weights = {k: getattr(model, k) for k in vars(model) if k.startswith(('W', 'U', 'b'))}
                np.savez(os.path.join(WEIGHTS_DIR, f'{args.model}_weights.npz'), **model_weights)
                np.savez(os.path.join(WEIGHTS_DIR, f'{args.model}_clf.npz'), W_out=W_out, b_out=b_out)
                best_val_loss = val_acc
        # Save weights if requested
        if args.save:
            print("Saving model and classifier weights...")
            model_weights = {k: getattr(model, k) for k in vars(model) if k.startswith(('W', 'U', 'b'))}
            np.savez(os.path.join(WEIGHTS_DIR, f'{args.model}_weights.npz'), **model_weights)
            np.savez(os.path.join(WEIGHTS_DIR, f'{args.model}_clf.npz'), W_out=W_out, b_out=b_out)
    # Interactive user input for prediction
    print("\n--- Interactive Sentiment Prediction ---")
    while True:
        user_text = input("Enter a review (or 'quit' to exit): ")
        if user_text.strip().lower() == 'quit':
            break
        tokens = tokenize(user_text)
        encoded = [vocab.get(tok, vocab['<UNK>']) for tok in tokens[:MAX_LEN]]
        if len(encoded) < MAX_LEN:
            encoded += [vocab['<PAD>']] * (MAX_LEN - len(encoded))
        emb = embedding[np.array(encoded)]
        x_seq = emb.reshape(MAX_LEN, EMBEDDING_DIM, 1)
        if isinstance(model, (VanillaRNNCell, StackedRNN)):
            hs = model.forward(x_seq)
            h_last = hs[-1] if isinstance(hs, list) else hs[-1]
        else:
            hs, cs = model.forward(x_seq)
            h_last = hs[-1]
        logits = np.dot(W_out, h_last) + b_out
        logits -= np.max(logits)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)
        print(f"[DEBUG] logits: {logits.ravel()}")
        print(f"[DEBUG] probs: {probs.ravel()}")
        pred = int(np.argmax(probs))
        print(f"Predicted sentiment: {'positive' if pred == 1 else 'negative'} (confidence: {float(np.max(probs)):.2f})\n")

    print("\nTip: If accuracy is stuck, try running with --model lstm --num_layers 2 or --num_layers 1 for both models.\n")

    # Example: Load data
    # imdb_data = load_imdb('data/reviews.csv')
    # tweets_data = load_tweets('data/tweets.csv')
    # quotes_data = load_quotes('data/book_quotes.txt')
    # TODO: Add training and evaluation routines
    pass 