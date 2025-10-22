import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Input, Conv1D, BatchNormalization, MaxPooling1D, 
                                   GlobalAveragePooling1D, Dense, Lambda, Dropout,
                                   Conv1DTranspose, UpSampling1D, Reshape, Flatten)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tensorflow.keras.regularizers import l2
import argparse
import h5py
import os
from sklearn.model_selection import train_test_split

# ==================== CONSTANTS ====================
# Triplet Loss Parameters
ALPHA = 0.3  # Triplet loss threshold/margin

# Training Parameters
LEARNING_RATE = 0.5e-2
LR_DROP = 0.5  # Learning rate multiplier every LR_DROP_INTERVAL
LR_DROP_INTERVAL = 20  # How many epochs to run before dropping learning rate
EPOCHS = 1000
BATCH_SIZE = 110
REG_WEIGHT = 0.0  # Regularization multiplier

# Model Architecture Parameters
EMB_SIZE = 40  # Embedding size
INPUT_SHAPE = (18, 5)  # Adjust based on your example_length

# ==================== LOSS FUNCTIONS ====================
def _triplet_distance_euclidean(tensors):
    """Compute Euclidean distance for triplet loss with margin alpha - EXACT PAPER VERSION"""
    anchor, positive, negative = tensors
    
    # Calculate squared Euclidean distances (as in the paper)
    pos_dist_sq = K.sum(K.square(anchor - positive), axis=-1)  # ||Ae - Pe||²
    neg_dist_sq = K.sum(K.square(anchor - negative), axis=-1)  # ||Ae - Ne||²
    
    # Apply triplet loss formula: max(||Ae-Pe||² - ||Ae-Ne||² + alpha, 0)
    return K.maximum(pos_dist_sq - neg_dist_sq + ALPHA, 0.0)
def _triplet_distance_cosine(tensors):
    """Compute cosine distance for triplet loss with margin alpha"""
    anchor, positive, negative = tensors
    pos_dist = 1 - K.sum(anchor * positive, axis=-1)  # Cosine distance: 1 - cosine_similarity
    neg_dist = 1 - K.sum(anchor * negative, axis=-1)
    return K.maximum(pos_dist - neg_dist + ALPHA, 0.0)

# ==================== LEARNING RATE SCHEDULER ====================
def lr_scheduler(epoch, lr):
    """Learning rate scheduler that drops LR every LR_DROP_INTERVAL epochs"""
    if epoch > 0 and epoch % LR_DROP_INTERVAL == 0:
        return lr * LR_DROP
    return lr

# ==================== MODEL BUILDING ====================
def build_encoder_model(input_shape):
    """
    Builds a CNN encoder model for keystroke data with regularization
    """
    print(f"Building encoder model for input shape: {input_shape}")
    
    x0 = Input(input_shape, name='encoder_input')
    x = x0

    # CNN architecture with regularization
    x = Conv1D(32, 3, padding='same', activation='relu', 
               kernel_regularizer=l2(REG_WEIGHT))(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    
    x = Conv1D(64, 3, padding='same', activation='relu',
               kernel_regularizer=l2(REG_WEIGHT))(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    
    x = Conv1D(128, 3, padding='same', activation='relu',
               kernel_regularizer=l2(REG_WEIGHT))(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    
    # Dense layers for feature extraction with regularization
    x = Dense(128, activation='relu', 
              kernel_regularizer=l2(REG_WEIGHT))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(64, activation='relu',
              kernel_regularizer=l2(REG_WEIGHT))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Final embedding
    y = Dense(EMB_SIZE, name='embedding',
              kernel_regularizer=l2(REG_WEIGHT))(x)
    y = Lambda(lambda x: K.l2_normalize(x, axis=1), name='l2_normalization')(y)

    encoder = Model(inputs=x0, outputs=y, name='encoder')
    encoder.summary()

    return encoder
def build_decoder_model(embedding_size, output_shape):
    """
    Builds a decoder model that reconstructs keystroke data from embeddings
    """
    print(f"Building decoder model for embedding size: {embedding_size}, output shape: {output_shape}")
    
    x0 = Input((embedding_size,), name='decoder_input')
    x = x0
    
    # Expand to original feature space with regularization
    x = Dense(64, activation='relu', 
              kernel_regularizer=l2(REG_WEIGHT))(x)
    x = BatchNormalization()(x)
    
    x = Dense(128, activation='relu',
              kernel_regularizer=l2(REG_WEIGHT))(x)
    x = BatchNormalization()(x)
    
    # Reshape for convolutional layers
    x = Dense(output_shape[0] * output_shape[1] // 4, activation='relu',
              kernel_regularizer=l2(REG_WEIGHT))(x)
    x = Reshape((output_shape[0] // 4, output_shape[1]))(x)
    
    # Transposed convolutions for reconstruction
    x = Conv1DTranspose(64, 3, padding='same', activation='relu',
                        kernel_regularizer=l2(REG_WEIGHT))(x)
    x = BatchNormalization()(x)
    x = UpSampling1D(2)(x)
    
    x = Conv1DTranspose(32, 3, padding='same', activation='relu',
                        kernel_regularizer=l2(REG_WEIGHT))(x)
    x = BatchNormalization()(x)
    x = UpSampling1D(2)(x)
    
    # Final reconstruction layer
    y = Conv1DTranspose(output_shape[1], 3, padding='same', 
                        activation='linear', name='decoder_output',
                        kernel_regularizer=l2(REG_WEIGHT))(x)

    decoder = Model(inputs=x0, outputs=y, name='decoder')
    decoder.summary()

    return decoder
def build_autoencoder_model(encoder, decoder):
    """Build autoencoder by combining encoder and decoder"""
    autoencoder_input = Input(encoder.input_shape[1:], name='autoencoder_input')
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    
    autoencoder = Model(inputs=autoencoder_input, outputs=decoded, name='autoencoder')
    autoencoder.summary()
    
    return autoencoder
def build_triplet_model(encoder, loss_type='euclidean'):
    """
    Builds a model that takes a triplet as input for contrastive learning
    """
    input_shape = encoder.input_shape[1:]
    
    input_A = Input(input_shape, name='anchor_input')
    input_B = Input(input_shape, name='positive_input')
    input_C = Input(input_shape, name='negative_input')

    x_A = encoder(input_A)
    x_B = encoder(input_B)
    x_C = encoder(input_C)

    # Choose loss function
    if loss_type == 'euclidean':
        triplet_loss = Lambda(_triplet_distance_euclidean, output_shape=(1,), name='triplet_loss')([x_A, x_B, x_C])
    else:  # cosine
        triplet_loss = Lambda(_triplet_distance_cosine, output_shape=(1,), name='triplet_loss')([x_A, x_B, x_C])

    triplet_model = Model([input_A, input_B, input_C], triplet_loss, name='triplet_model')
    
    # Compile with custom learning rate
    optimizer = Adam(learning_rate=LEARNING_RATE)
    triplet_model.compile(optimizer=optimizer, loss='mse')
    
    return triplet_model
def build_verification_nn(input_dim=EMB_SIZE):
    """
    Build Neural Network that takes embedding differences and predicts same/different user
    Input: Difference between two embeddings (40-dimensional)
    Output: Probability (0-1) that they're from same user
    """
    print("Building Verification Neural Network...")
    
    # Input: difference between two embeddings
    input_layer = Input(shape=(input_dim,), name='embedding_diff_input')
    
    x = input_layer
    
    # Neural network architecture
    x = Dense(64, activation='relu', kernel_regularizer=l2(REG_WEIGHT))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(32, activation='relu', kernel_regularizer=l2(REG_WEIGHT))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(16, activation='relu', kernel_regularizer=l2(REG_WEIGHT))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Output: probability of same user (0-1)
    output_layer = Dense(1, activation='sigmoid', name='verification_output')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer, name='verification_nn')
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    return model

# ==================== DATA LOADING ====================
def load_triplet_data(hdf5_path):
    """Load triplet data from HDF5 file"""
    with h5py.File(hdf5_path, 'r') as f:
        anchors = f['X_train_anchors'][:]
        positives = f['X_train_positives'][:]
        negatives = f['X_train_negatives'][:]
        
        # Create dummy labels (triplet loss doesn't need them)
        dummy_labels = np.zeros(len(anchors))
        
    return [anchors, positives, negatives], dummy_labels
def create_nn_training_data(encoder, hdf5_path):
    """Create training data for Neural Network from embedding differences"""
    with h5py.File(hdf5_path, 'r') as f:
        # Get embeddings for all users
        user_embeddings = {}
        
        for i in range(99):  # For each user
            if f'X_test_{i}' in f:
                user_data = f[f'X_test_{i}'][:]
                if len(user_data) > 0:
                    embeddings = encoder.predict(user_data, verbose=0, batch_size=BATCH_SIZE)
                    user_embeddings[i] = embeddings
        
        # Create positive and negative pairs with DIFFERENCES
        X_diff = []  # Element-wise differences between embeddings
        y_diff = []  # Labels (1 = same user, 0 = different users)
        
        # Positive pairs (same user) - both [A-P] and [P-A] differences
        print("Creating positive pairs...")
        for user_id, embeddings in user_embeddings.items():
            if len(embeddings) >= 2:
                for i in range(min(10, len(embeddings))):
                    for j in range(i+1, min(i+11, len(embeddings))):
                        # Both directions of difference (as mentioned in paper)
                        diff1 = embeddings[i] - embeddings[j]  # A - P
                        diff2 = embeddings[j] - embeddings[i]  # P - A
                        
                        X_diff.extend([diff1, diff2])
                        y_diff.extend([1, 1])  # Same user
        
        # Negative pairs (different users)
        print("Creating negative pairs...")
        user_ids = list(user_embeddings.keys())
        n_negative = len(X_diff)  # Balance the dataset
        
        for _ in range(n_negative):
            user1, user2 = np.random.choice(user_ids, 2, replace=False)
            if user1 in user_embeddings and user2 in user_embeddings:
                if len(user_embeddings[user1]) > 0 and len(user_embeddings[user2]) > 0:
                    idx1 = np.random.randint(len(user_embeddings[user1]))
                    idx2 = np.random.randint(len(user_embeddings[user2]))
                    
                    emb1 = user_embeddings[user1][idx1]
                    emb2 = user_embeddings[user2][idx2]
                    
                    diff = emb1 - emb2
                    X_diff.append(diff)
                    y_diff.append(0)  # Different users
        
        return np.array(X_diff), np.array(y_diff)

# ==================== TRAINING FUNCTIONS ====================
def train_encoder_triplet(hdf5_path, input_shape, epochs=EPOCHS, batch_size=BATCH_SIZE, loss_type='euclidean'):
    """Train encoder using triplet loss"""
    print("=== Training Encoder with Triplet Loss ===")
    print(f"Parameters: alpha={ALPHA}, lr={LEARNING_RATE}, batch_size={batch_size}, epochs={epochs}")
    
    encoder = build_encoder_model(input_shape)
    triplet_model = build_triplet_model(encoder, loss_type)
    
    # Load data
    X_triplet, y_triplet = load_triplet_data(hdf5_path)
    
    # Create a custom callback to save encoder weights
    class SaveEncoderWeights(tf.keras.callbacks.Callback):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            self.best_val_loss = float('inf')
            
        def on_epoch_end(self, epoch, logs=None):
            if logs['val_loss'] < self.best_val_loss:
                self.best_val_loss = logs['val_loss']
                self.encoder.save_weights('encoder_weights.weights.h5')
                print(f"Saved best encoder weights with val_loss: {self.best_val_loss:.4f}")
    
    # Callbacks
    callbacks = [
        EarlyStopping(patience=20, restore_best_weights=True, min_delta=1e-4),
        SaveEncoderWeights(encoder),
        LearningRateScheduler(lr_scheduler)
    ]
    
    # Train
    history = triplet_model.fit(
        X_triplet, y_triplet,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )
    
    # Load best weights
    encoder.load_weights('encoder_weights.weights.h5')
    
    return encoder, history
def train_autoencoder(hdf5_path, input_shape, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """Train autoencoder (encoder + decoder)"""
    print("=== Training Autoencoder ===")
    
    encoder = build_encoder_model(input_shape)
    decoder = build_decoder_model(EMB_SIZE, input_shape)
    autoencoder = build_autoencoder_model(encoder, decoder)
    
    optimizer = Adam(learning_rate=LEARNING_RATE)
    autoencoder.compile(optimizer=optimizer, loss='mse')
    
    # Load training data
    with h5py.File(hdf5_path, 'r') as f:
        X_train = f['X_train'][:]
    
    # Callbacks
    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True),
        ModelCheckpoint('autoencoder_weights.weights.h5', save_best_only=True, save_weights_only=True),
        LearningRateScheduler(lr_scheduler)
    ]
    
    # Train
    history = autoencoder.fit(
        X_train, X_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )
    
    return encoder, decoder, history
def train_verification_nn(encoder, hdf5_path, epochs=EPOCHS//2, batch_size=BATCH_SIZE):
    """Train Neural Network verification model on embedding differences"""
    print("=== Training Neural Network Verification Model ===")
    
    # Create training data
    X_diff, y_diff = create_nn_training_data(encoder, hdf5_path)
    print(f"Training data shape: {X_diff.shape}")
    print(f"Class distribution: {np.bincount(y_diff)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_diff, y_diff, test_size=0.2, random_state=42, stratify=y_diff
    )
    
    # Build and train NN
    verification_nn = build_verification_nn(input_dim=X_diff.shape[1])
    
    # Callbacks
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ModelCheckpoint('verification_nn_weights.weights.h5', save_best_only=True, save_weights_only=True),
        LearningRateScheduler(lr_scheduler)
    ]
    
    history = verification_nn.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_acc = verification_nn.evaluate(X_test, y_test, verbose=0)
    print(f"Final Test Accuracy: {test_acc:.4f}")
    
    return verification_nn, history

# ==================== EVALUATION METRICS ====================
def calculate_far_frr(encoder, verification_nn, hdf5_path, num_test_samples=1000, num_references=1, user_type='all'):
    """
    Calculate False Acceptance Rate (FAR) and False Rejection Rate (FRR)
    with configurable number of references and user types
    
    Parameters:
    - num_references: 1 or 5 (uses majority voting for 5)
    - user_type: 'seen' (training users), 'unseen' (test users), 'all'
    """
    print("=== Calculating FAR and FRR ===")
    print(f"Configuration: {num_references} reference(s), users: {user_type}")
    
    with h5py.File(hdf5_path, 'r') as f:
        # Get user splits from the HDF5 file structure
        if user_type == 'seen':
            # Use training users (first 80% of users)
            user_range = range(0, 79)  # Adjust based on your actual split
        elif user_type == 'unseen':
            # Use test users (last 20% of users)  
            user_range = range(80, 99)  # Adjust based on your actual split
        else:  # 'all'
            user_range = range(99)
        
        # Get embeddings for selected users
        user_embeddings = {}
        for i in user_range:
            dataset_name = f'X_test_{i}'
            if dataset_name in f:
                user_data = f[dataset_name][:]
                if len(user_data) > 0:
                    embeddings = encoder.predict(user_data, verbose=0, batch_size=BATCH_SIZE)
                    user_embeddings[i] = embeddings
    
    # Generate test pairs
    genuine_scores = []   # Scores for same-user pairs
    impostor_scores = []  # Scores for different-user pairs
    
    user_ids = list(user_embeddings.keys())
    
    # Genuine pairs (same user) - with multiple references support
    for user_id in user_ids:
        embeddings = user_embeddings[user_id]
        if len(embeddings) >= max(2, num_references + 1):
            for _ in range(min(20, len(embeddings) // (num_references + 1))):
                # Select query sample and reference samples
                available_indices = list(range(len(embeddings)))
                query_idx = np.random.choice(available_indices)
                available_indices.remove(query_idx)
                
                if num_references == 1:
                    # Single reference
                    ref_idx = np.random.choice(available_indices)
                    diff = embeddings[query_idx] - embeddings[ref_idx]
                    score = verification_nn.predict(diff.reshape(1, -1), verbose=0)[0][0]
                    genuine_scores.append(score)
                else:
                    # Multiple references with majority voting (5 references)
                    ref_indices = np.random.choice(available_indices, min(num_references, len(available_indices)), replace=False)
                    ref_scores = []
                    for ref_idx in ref_indices:
                        diff = embeddings[query_idx] - embeddings[ref_idx]
                        score = verification_nn.predict(diff.reshape(1, -1), verbose=0)[0][0]
                        ref_scores.append(1 if score > 0.5 else 0)  # Convert to binary
                    
                    # Majority vote
                    majority_vote = 1 if sum(ref_scores) > len(ref_scores) / 2 else 0
                    genuine_scores.append(majority_vote)
    
    # Impostor pairs (different users) - with multiple references support
    for _ in range(len(genuine_scores)):  # Same number as genuine pairs
        user1, user2 = np.random.choice(user_ids, 2, replace=False)
        if user1 in user_embeddings and user2 in user_embeddings:
            if len(user_embeddings[user1]) > 0 and len(user_embeddings[user2]) > 0:
                if num_references == 1:
                    # Single reference
                    query_emb = user_embeddings[user1][np.random.randint(len(user_embeddings[user1]))]
                    ref_emb = user_embeddings[user2][np.random.randint(len(user_embeddings[user2]))]
                    diff = query_emb - ref_emb
                    score = verification_nn.predict(diff.reshape(1, -1), verbose=0)[0][0]
                    impostor_scores.append(score)
                else:
                    # Multiple references with majority voting
                    query_emb = user_embeddings[user1][np.random.randint(len(user_embeddings[user1]))]
                    ref_indices = np.random.choice(len(user_embeddings[user2]), min(num_references, len(user_embeddings[user2])), replace=False)
                    ref_scores = []
                    for ref_idx in ref_indices:
                        ref_emb = user_embeddings[user2][ref_idx]
                        diff = query_emb - ref_emb
                        score = verification_nn.predict(diff.reshape(1, -1), verbose=0)[0][0]
                        ref_scores.append(1 if score > 0.5 else 0)
                    
                    # Majority vote
                    majority_vote = 1 if sum(ref_scores) > len(ref_scores) / 2 else 0
                    impostor_scores.append(majority_vote)
    
    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)
    
    print(f"Genuine pairs: {len(genuine_scores)}, Impostor pairs: {len(impostor_scores)}")
    
    # For majority voting, scores are already binary (0 or 1)
    if num_references > 1:
        # Calculate binary classification metrics directly
        far = np.mean(impostor_scores)  # Impostors accepted (score = 1)
        frr = 1 - np.mean(genuine_scores)  # Genuine rejected (score = 0)
        
        print(f"\n=== FAR/FRR Results (Majority Voting) ===")
        print(f"FAR = {far:.4f}, FRR = {frr:.4f}")
        print(f"Genuine acceptance rate: {np.mean(genuine_scores):.4f}")
        print(f"Impostor rejection rate: {1 - np.mean(impostor_scores):.4f}")
        
        return [far], [frr], [0.5], far  # EER approximation
    
    else:
        # Single reference - use threshold sweep
        thresholds = np.linspace(0, 1, 100)
        far_values = []
        frr_values = []
        
        for threshold in thresholds:
            far = np.mean(impostor_scores >= threshold)
            frr = np.mean(genuine_scores < threshold)
            far_values.append(far)
            frr_values.append(frr)
        
        # Find EER
        eer_threshold = thresholds[np.argmin(np.abs(np.array(far_values) - np.array(frr_values)))]
        eer = (far_values[np.argmin(np.abs(np.array(far_values) - np.array(frr_values)))] + 
               frr_values[np.argmin(np.abs(np.array(far_values) - np.array(frr_values)))]) / 2
        
        # Results at threshold 0.5
        far_05 = np.mean(impostor_scores >= 0.5)
        frr_05 = np.mean(genuine_scores < 0.5)
        
        print(f"\n=== FAR/FRR Results ===")
        print(f"Threshold 0.5: FAR = {far_05:.4f}, FRR = {frr_05:.4f}")
        print(f"EER (Equal Error Rate): {eer:.4f} at threshold {eer_threshold:.3f}")
        print(f"Genuine scores mean: {np.mean(genuine_scores):.4f} ± {np.std(genuine_scores):.4f}")
        print(f"Impostor scores mean: {np.mean(impostor_scores):.4f} ± {np.std(impostor_scores):.4f}")
        
        return far_values, frr_values, thresholds, eer
def plot_far_frr(far_values, frr_values, thresholds, eer):
    """Plot FAR and FRR curves (optional)"""
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, far_values, 'b-', label='FAR (False Acceptance Rate)')
        plt.plot(thresholds, frr_values, 'r-', label='FRR (False Rejection Rate)')
        plt.axvline(x=0.5, color='gray', linestyle='--', label='Threshold 0.5')
        plt.xlabel('Decision Threshold')
        plt.ylabel('Rate')
        plt.title(f'FAR and FRR Curves (EER = {eer:.4f})')
        plt.legend()
        plt.grid(True)
        plt.savefig('far_frr_curve.png')
        plt.close()
        print("FAR/FRR curve saved as 'far_frr_curve.png'")
        
    except ImportError:
        print("Matplotlib not available. Skipping plot.")

# ==================== MAIN FUNCTION ====================
def main():
    parser = argparse.ArgumentParser(description='Train keystroke verification system')
    parser.add_argument('--mode', choices=['encoder', 'decoder', 'both', 'autoencoder', 'verification_nn', 'evaluate'], 
                       default='both', help='What to train')
    parser.add_argument('--hdf5_path', default='99_users_mixed_80_10_10.hdf5', 
                       help='Path to HDF5 data file')
    parser.add_argument('--encoder_weights', help='Path to encoder weights')
    parser.add_argument('--verification_weights', help='Path to verification model weights')
    parser.add_argument('--loss_type', choices=['euclidean', 'cosine'], default='euclidean',
                       help='Type of distance metric for triplet loss')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--eval_far_frr', action='store_true', help='Calculate FAR/FRR after training')
    parser.add_argument('--num_references', type=int, default=1, help='Number of reference samples (1 or 5)')
    parser.add_argument('--users', choices=['seen', 'unseen', 'all'], default='all',
                       help='Which users to evaluate on: seen (training), unseen (test), or all')
    
    args = parser.parse_args()
    
    print(f"=== Training Parameters ===")
    print(f"Alpha (margin): {ALPHA}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"LR drop: {LR_DROP} every {LR_DROP_INTERVAL} epochs")
    print(f"Embedding size: {EMB_SIZE}")
    print(f"Regularization weight: {REG_WEIGHT}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Loss type: {args.loss_type}")
    print(f"Number of references: {args.num_references}")
    print(f"Users to evaluate: {args.users}")
    
    if args.mode == 'encoder':
        # Train only encoder with triplet loss
        encoder, _ = train_encoder_triplet(args.hdf5_path, INPUT_SHAPE, args.epochs, args.batch_size, args.loss_type)
        encoder.save('trained_encoder.h5')
        print("Encoder training completed and saved!")
        
    elif args.mode == 'decoder':
        # Train only verification model (decoder)
        if not args.encoder_weights:
            raise ValueError("Encoder weights path required for decoder-only training")
        
        encoder = build_encoder_model(INPUT_SHAPE)
        encoder.load_weights(args.encoder_weights)
        verification_model, _ = train_verification_nn(encoder, args.hdf5_path, args.epochs//2, args.batch_size)
        verification_model.save('trained_verification_nn.h5')
        print("Neural Network verification model training completed and saved!")
        
    elif args.mode == 'both':
        # Train both encoder and decoder
        encoder, _ = train_encoder_triplet(args.hdf5_path, INPUT_SHAPE, args.epochs, args.batch_size, args.loss_type)
        verification_model, _ = train_verification_nn(encoder, args.hdf5_path, args.epochs//2, args.batch_size)
        
        encoder.save('trained_encoder.h5')
        verification_model.save('trained_verification_nn.h5')
        print("Both encoder and Neural Network verification model training completed and saved!")
        
        # Evaluate FAR/FRR if requested
        if args.eval_far_frr:
            far, frr, thresholds, eer = calculate_far_frr(encoder, verification_model, args.hdf5_path, 
                                                         num_references=args.num_references, user_type=args.users)
            plot_far_frr(far, frr, thresholds, eer)
        
    elif args.mode == 'autoencoder':
        # Train autoencoder
        encoder, decoder, _ = train_autoencoder(args.hdf5_path, INPUT_SHAPE, args.epochs, args.batch_size)
        encoder.save('trained_encoder_ae.h5')
        decoder.save('trained_decoder_ae.h5')
        print("Autoencoder training completed and saved!")
        
    elif args.mode == 'verification_nn':
        # Train only the Neural Network verification model
        if not args.encoder_weights:
            raise ValueError("Encoder weights path required for verification_nn training")
        
        encoder = build_encoder_model(INPUT_SHAPE)
        encoder.load_weights(args.encoder_weights)
        verification_model, _ = train_verification_nn(encoder, args.hdf5_path, args.epochs, args.batch_size)
        verification_model.save('trained_verification_nn.h5')
        print("Neural Network verification model training completed and saved!")
        
    elif args.mode == 'evaluate':
        # Only evaluate existing models
        if not args.encoder_weights or not args.verification_weights:
            raise ValueError("Both encoder and verification weights required for evaluation")
        
        encoder = build_encoder_model(INPUT_SHAPE)
        encoder.load_weights(args.encoder_weights)
        
        verification_model = build_verification_nn()
        verification_model.load_weights(args.verification_weights)
        
        far, frr, thresholds, eer = calculate_far_frr(encoder, verification_model, args.hdf5_path,
                                                     num_references=args.num_references, user_type=args.users)
        plot_far_frr(far, frr, thresholds, eer)


if __name__ == "__main__":
    main()
