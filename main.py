import tensorflow as tf
from model import SimpleTransformer
import numpy as np
import PyPDF2
import re
import pickle
import os
from sklearn.model_selection import KFold

# ------------------------------
# PDF reading and preprocessing
# ------------------------------
def read_pdf(pdf_path):
    """
    Extracts and processes text from a PDF file.
    
    How it works:
    1. Opens a PDF file using PyPDF2 library
    2. Reads through each page of the PDF
    3. Extracts raw text content
    4. Cleans the text by:
       - Removing extra whitespace
       - Removing special characters
       - Standardizing spacing
    
    Args:
        pdf_path (str): Path to the PDF file to read
        
    Returns:
        str: Cleaned text content from the PDF
    """
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    # Clean text
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text.strip()

def preprocess_text(text, vocab_size=1000):
    """
    Converts raw text into a format suitable for the language model.
    
    How it works:
    1. Creates a tokenizer that will:
       - Convert words to numbers (e.g., "whale" -> 42)
       - Handle unknown words with <OOV> (Out Of Vocabulary) token
       - Limit vocabulary to most common words
    2. Processes the text by:
       - Converting to lowercase
       - Splitting into individual words
       - Creating a dictionary of word->number mappings
       - Converting the text into sequences of numbers
    
    Args:
        text (str): Raw text to process
        vocab_size (int): Maximum number of unique words to keep
        
    Returns:
        tuple: (sequences of numbers, tokenizer object)
    """
    # Add special tokens
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=vocab_size,
        oov_token="<OOV>",
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    )
    words = text.lower().split()
    tokenizer.fit_on_texts([' '.join(words)])
    sequences = tokenizer.texts_to_sequences([' '.join(words)])[0]
    return sequences, tokenizer

def create_training_data(sequences, seq_length=50):
    """
    Creates input-output pairs for training the language model.
    
    How it works:
    1. Takes a sequence of numbers (representing words)
    2. Creates sliding windows of text, where:
       - Input: Words 1-50
       - Target: Words 2-51
       This teaches the model to predict the next word
    
    Example:
    Text: "the cat sat on mat"
    Becomes:
    Input: "the cat sat on"
    Target: "cat sat on mat"
    
    Args:
        sequences (list): List of numbers representing words
        seq_length (int): Length of each training sequence
        
    Returns:
        tuple: (input sequences, target sequences) as numpy arrays
    """
    input_sequences = []
    for i in range(len(sequences) - seq_length):
        input_sequences.append(sequences[i:i + seq_length + 1])
    x = np.array([seq[:-1] for seq in input_sequences])
    y = np.array([seq[1:] for seq in input_sequences])
    return x, y

# ------------------------------
# Main function with training
# ------------------------------
def main():
    # Directories
    save_dir = "saved_model"
    os.makedirs(save_dir, exist_ok=True)
    pdf_path = "mobydickpdf.pdf"

    # Load PDF
    try:
        text = read_pdf(pdf_path)
        print(f"Loaded PDF. Text length: {len(text)} characters")
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return

    # Model / training parameters
    vocab_size = 10000
    seq_length = 50
    d_model = 128
    num_heads = 4
    num_layers = 3
    batch_size = 32
    epochs = 50  # early stopping will handle real stopping

    # Preprocess
    sequences, tokenizer = preprocess_text(text, vocab_size)
    x_train, y_train = create_training_data(sequences, seq_length)
    print(f"Training data shape: {x_train.shape}, Vocabulary size: {len(tokenizer.word_index)}")

    # ------------------------------
    # Callbacks
    # ------------------------------
    checkpoint_path = os.path.join(save_dir, "model_checkpoint.keras")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=False,
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    callbacks = [checkpoint_callback, early_stop, lr_callback]

    # ------------------------------
    # Optional: Cross-validation
    # ------------------------------
    k_folds = 3  # 3-fold CV for demonstration
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_no = 1
    for train_index, val_index in kf.split(x_train):
        print(f"\nTraining fold {fold_no}/{k_folds}")
        x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        # Create fresh model for each fold
        model = SimpleTransformer(vocab_size, d_model, num_heads, num_layers)
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        # Train
        model.fit(
            x_train_fold, y_train_fold,
            validation_data=(x_val_fold, y_val_fold),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        fold_no += 1

    # ------------------------------
    # Save final model
    # ------------------------------
    final_model_path = os.path.join(save_dir, "final_model.keras")
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")

    # Save tokenizer
    tokenizer_path = os.path.join(save_dir, "tokenizer.pickle")
    with open(tokenizer_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Tokenizer saved to {tokenizer_path}")

if __name__ == "__main__":
    main()
