import os
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import re
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

def get_dataset():
    """Returns a conversational dataset of user queries and chatbot responses."""
    return [
        ("hello", "hi there ! how can i help you today ?"),
        ("who are you", "i am a neural chatbot built using tensorflow ."),
        ("what can you do", "i can chat with you and answer questions ."),
        ("how are you", "i am doing great , thank you for asking !"),
        ("good morning", "good morning ! hope you have a wonderful day ."),
        ("goodbye", "goodbye ! have a nice day ."),
        ("tell me a joke", "why did the computer go to the doctor ? it had a virus !"),
        ("who created you", "i was created by a Gideon Mwiti using deep learning ."),
        ("what is deep learning", "it is a branch of machine learning based on neural networks ."),
        ("what is tensorflow", "tensorflow is an open source machine learning framework by google ."),
        ("what is attention", "attention helps models focus on relevant parts of the input ."),
        ("thank you", "you are very welcome !"),
        ("awesome", "thanks ! i try my best ."),
        ("help me", "sure ! please tell me what you need help with ."),
        ("what is your name", "you can call me chatty !"),
        ("where do you live", "i live in the cloud ."),
        ("how old are you", "i was recently compiled , so i am very young !"),
        ("are you human", "no , i am an artificial intelligence chatbot .")
    ]

def clean_and_tokenize(sentence):
    """Tokenizes a sentence by separating punctuation and mapping to lowercase."""
    s = re.sub(r"([?.!,¿])", r" \1 ", sentence.lower())
    s = re.sub(r'[" "]+', " ", s)
    return s.strip().split()

def main():
    print("====================================================")
    print("Project 41: Seq2Seq Chatbot with Attention Mechanism")
    print("Goal: Generate context-aware replies using Additive Attention")
    print("====================================================\n")

    # 1. Load Dataset
    pairs = get_dataset()
    print(f"Loaded {len(pairs)} dialogue pairs.")

    # 2. Build Shared Vocabulary
    # Chatbots use a single shared vocabulary since queries and responses are in the same language
    word_vocab = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
    for q, r in pairs:
        for word in clean_and_tokenize(q):
            if word not in word_vocab:
                word_vocab[word] = len(word_vocab)
        for word in clean_and_tokenize(r):
            if word not in word_vocab:
                word_vocab[word] = len(word_vocab)

    vocab_size = len(word_vocab)
    id_to_word = {i: w for w, i in word_vocab.items()}
    print(f"Shared Vocabulary Size: {vocab_size} unique tokens.")

    # Max Sequence Lengths
    max_enc_len = max(len(clean_and_tokenize(q)) for q, _ in pairs)
    max_dec_len = max(len(clean_and_tokenize(r)) for _, r in pairs) + 2 # Add <start> and <end>
    print(f"Max Encoder Sequence Length (Query): {max_enc_len}")
    print(f"Max Decoder Sequence Length (Reply): {max_dec_len}")

    # 3. Create Training Tensors
    num_samples = len(pairs)
    encoder_input_data = np.zeros((num_samples, max_enc_len), dtype=np.int32)
    decoder_input_data = np.zeros((num_samples, max_dec_len), dtype=np.int32)
    decoder_target_data = np.zeros((num_samples, max_dec_len), dtype=np.int32)

    for idx, (q, r) in enumerate(pairs):
        # Encoder Input
        q_tokens = clean_and_tokenize(q)
        for t, token in enumerate(q_tokens):
            encoder_input_data[idx, t] = word_vocab[token]

        # Decoder Input and Target
        r_tokens = clean_and_tokenize(r)
        decoder_input_data[idx, 0] = word_vocab["<start>"]
        for t, token in enumerate(r_tokens):
            decoder_input_data[idx, t + 1] = word_vocab[token]
            decoder_target_data[idx, t] = word_vocab[token]
        decoder_target_data[idx, len(r_tokens)] = word_vocab["<end>"]

    # 4. Build Seq2Seq + Attention Model Architecture
    latent_dim = 64

    # --- Encoder ---
    encoder_inputs = layers.Input(shape=(max_enc_len,), name="encoder_inputs")
    # Shared embedding space with mask_zero=True
    shared_embedding = layers.Embedding(input_dim=vocab_size, output_dim=32, mask_zero=True, name="shared_embedding")
    encoder_emb = shared_embedding(encoder_inputs)
    
    # Bidirectional LSTM to capture bidirectional contextual information
    encoder_lstm = layers.Bidirectional(layers.LSTM(latent_dim, return_sequences=True, return_state=True, name="encoder_lstm"))
    encoder_outputs, state_f_h, state_f_c, state_b_h, state_b_c = encoder_lstm(encoder_emb)
    
    # Concatenate forward and backward final states
    state_h = layers.Concatenate()([state_f_h, state_b_h])
    state_c = layers.Concatenate()([state_f_c, state_b_c])
    encoder_states = [state_h, state_c] # concatenated shape: (latent_dim * 2,)

    # --- Decoder ---
    decoder_inputs = layers.Input(shape=(max_dec_len,), name="decoder_inputs")
    decoder_emb = shared_embedding(decoder_inputs)
    # Decoder LSTM units must equal concatenated encoder states: latent_dim * 2
    decoder_lstm = layers.LSTM(latent_dim * 2, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(decoder_emb, initial_state=encoder_states)

    # --- Additive Attention (Bahdanau-style) ---
    attention_layer = layers.AdditiveAttention(name="attention_layer")
    # Computes weights between query (decoder_outputs) and values (encoder_outputs)
    context_vector = attention_layer([decoder_outputs, encoder_outputs])

    # Concatenate Attention context vector with Decoder outputs
    decoder_combined = layers.Concatenate(axis=-1)([decoder_outputs, context_vector])

    # Projection layer mapping to vocab
    decoder_dense = layers.Dense(vocab_size, activation="softmax", name="decoder_dense")
    output_predictions = decoder_dense(decoder_combined)

    # Training Model
    model = tf.keras.Model([encoder_inputs, decoder_inputs], output_predictions)
    model.summary()

    # 5. Compile and Train
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.008),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    epochs = 220
    batch_size = 8
    print(f"\nTraining chatbot model (Epochs: {epochs}, Batch Size: {batch_size})...")
    history = model.fit(
        [encoder_input_data, decoder_input_data], decoder_target_data,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # 6. Build Chatbot Inference Models
    print("\nBuilding chatbot inference models...")
    # Encoder Inference Model
    encoder_model = tf.keras.Model(encoder_inputs, [encoder_outputs] + encoder_states)

    # Decoder Inference Model
    # Expects single word ID, full encoder outputs (keys/values), and initial LSTM states
    decoder_state_input_h = layers.Input(shape=(latent_dim * 2,), name="input_state_h")
    decoder_state_input_c = layers.Input(shape=(latent_dim * 2,), name="input_state_c")
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    encoder_outputs_input = layers.Input(shape=(max_enc_len, latent_dim * 2), name="encoder_outputs_input")

    decoder_inputs_single = layers.Input(shape=(1,), name="input_single_word")
    decoder_embedded_single = shared_embedding(decoder_inputs_single)

    decoder_outputs_single, state_h_single, state_c_single = decoder_lstm(
        decoder_embedded_single, initial_state=decoder_states_inputs
    )
    decoder_states_outputs_single = [state_h_single, state_c_single]

    context_vector_single = attention_layer([decoder_outputs_single, encoder_outputs_input])
    decoder_combined_single = layers.Concatenate(axis=-1)([decoder_outputs_single, context_vector_single])
    decoder_outputs_single = decoder_dense(decoder_combined_single)

    decoder_model = tf.keras.Model(
        [decoder_inputs_single, encoder_outputs_input] + decoder_states_inputs,
        [decoder_outputs_single] + decoder_states_outputs_single
    )

    def chatbot_respond(query_sentence):
        """Generates a reply for a user query using greedy decoding and attention mapping."""
        tokens = clean_and_tokenize(query_sentence)
        input_seq = np.zeros((1, max_enc_len), dtype=np.int32)
        for t, token in enumerate(tokens):
            if t < max_enc_len:
                input_seq[0, t] = word_vocab.get(token, word_vocab["<unk>"])

        # Run Encoder to get outputs and initial state
        enc_out, state_h_val, state_c_val = encoder_model.predict(input_seq, verbose=0)
        states_value = [state_h_val, state_c_val]

        # Initial Decoder target sequence is '<start>' (1)
        target_seq = np.zeros((1, 1), dtype=np.int32)
        target_seq[0, 0] = word_vocab["<start>"]

        decoded_words = []
        stop_condition = False

        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq, enc_out] + states_value, verbose=0)

            # Sample predicted token (greedy decoding)
            sampled_token_idx = np.argmax(output_tokens[0, 0, :])
            sampled_word = id_to_word.get(sampled_token_idx, "<unk>")

            if sampled_word == "<end>" or len(decoded_words) >= max_dec_len:
                stop_condition = True
            elif sampled_word == "<pad>":
                stop_condition = True
            else:
                decoded_words.append(sampled_word)

            # Feed the generated word ID as next input
            target_seq = np.zeros((1, 1), dtype=np.int32)
            target_seq[0, 0] = sampled_token_idx
            states_value = [h, c]

        return " ".join(decoded_words)

    # 7. Evaluate on Conversational Test Cases
    test_queries = [
        "hello",
        "who are you",
        "what is tensorflow",
        "what is attention",
        "goodbye"
    ]

    print("\nEvaluating Chatbot Responses:")
    evaluation_logs = []
    for query in test_queries:
        reply = chatbot_respond(query)
        print(f"\nUser: {query}")
        print(f"Bot:  {reply}")
        evaluation_logs.append((query, reply))

    # 8. Save Metrics & Mock Chat UI Visualization
    print("\nGenerating visual chat panel...")
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.8])

    # Left Panel: Loss & Accuracy
    ax_metrics = fig.add_subplot(gs[0])
    ax_metrics.plot(history.history['loss'], label='Loss', color='#e74c3c', linewidth=2)
    ax_metrics.plot(history.history['accuracy'], label='Accuracy', color='#2ecc71', linewidth=2)
    ax_metrics.set_title("Training Loss & Accuracy Curves", fontsize=11, fontweight="bold", pad=10)
    ax_metrics.set_xlabel("Epochs")
    ax_metrics.set_ylabel("Metric Value")
    ax_metrics.legend(loc='center right')
    ax_metrics.grid(True, alpha=0.3)

    # Right Panel: Mock Chatbot UI Interface
    ax_chat = fig.add_subplot(gs[1])
    ax_chat.axis('off')
    ax_chat.set_xlim(0, 10)
    ax_chat.set_ylim(0, 10)

    # Draw Chat phone background container
    bg_rect = plt.Rectangle((0, 0), 10, 10, fill=True, color='#f4f6f7', edgecolor='#bdc3c7', lw=2.5, zorder=1)
    ax_chat.add_patch(bg_rect)

    # Draw Blue Header Bar
    header_rect = plt.Rectangle((0, 8.8), 10, 1.2, fill=True, color='#2980b9', zorder=2)
    ax_chat.add_patch(header_rect)
    ax_chat.text(5, 9.4, "Chatty AI Assistant (Seq2Seq + Attention)", color='white', 
                 fontsize=12, fontweight='bold', ha='center', va='center', zorder=3)

    # Display dialogue bubbles
    # User messages are right-aligned (blue bubbles)
    # Bot responses are left-aligned (light grey/white bubbles)
    y_coord = 7.7
    
    # We display up to 4 interactions
    for idx, (u, b) in enumerate(evaluation_logs[:4]):
        # User bubble
        user_text = f"User: {u}"
        ax_chat.text(9.6, y_coord, user_text, ha='right', va='center', fontsize=9.5, fontweight='semibold',
                     color='#2c3e50', zorder=3,
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='#d4efdf', edgecolor='#a9dfbf'))
        y_coord -= 0.75
        
        # Bot bubble
        bot_text = f"Bot: {b}"
        ax_chat.text(0.4, y_coord, bot_text, ha='left', va='center', fontsize=9.5, fontweight='semibold',
                     color='#2c3e50', zorder=3,
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffffff', edgecolor='#d5dbdb'))
        y_coord -= 1.05

    # Draw Bottom input box mockup
    footer_rect = plt.Rectangle((0, 0), 10, 1.0, fill=True, color='#eaeded', zorder=2)
    ax_chat.add_patch(footer_rect)
    ax_chat.text(0.5, 0.5, "Type a message...", color='#7f8c8d', fontsize=10, fontstyle='italic', va='center', zorder=3)
    
    # Draw send button
    send_rect = plt.Rectangle((8.2, 0.25), 1.5, 0.5, fill=True, color='#3498db', zorder=3)
    ax_chat.add_patch(send_rect)
    ax_chat.text(8.95, 0.5, "Send", color='white', fontsize=9.5, fontweight='bold', ha='center', va='center', zorder=4)

    plt.tight_layout()
    output_filename = "chatbot_attention_results.png"
    plt.savefig(output_filename, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"\nSuccess! Results panel saved as '{output_filename}'.")
    print("====================================================")

if __name__ == "__main__":
    main()
