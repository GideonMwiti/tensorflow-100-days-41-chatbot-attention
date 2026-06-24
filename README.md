# Project 41: Chatbot using Seq2Seq + Attention

This project builds a simple conversational chatbot utilizing a **Sequence-to-Sequence (Seq2Seq)** architecture enhanced with **Additive Attention (Bahdanau-style)** in TensorFlow 2. 

## Theoretical Overview

### 1. The Seq2Seq Context Compression Bottleneck
Standard Seq2Seq models compress the entire input sequence (encoder sequence) into a single fixed-size context vector (the final hidden states of the encoder LSTM). For long sequences, this creates a severe bottleneck, leading to information loss.

### 2. How Attention Solves the Bottleneck
Instead of compressing the input into a single vector, **Attention** allows the decoder to look back at the entire sequence of encoder outputs at each decoding step:
* **Keys & Values**: The encoder outputs at each input token step.
* **Query**: The current decoder LSTM hidden state.
* **Alignment Scores**: Calculated using a small feed-forward network (in Additive Attention) to measure similarity between the query and each key:
$$\text{score}(q, k) = v_a^T \tanh(W_a q + U_a k)$$
* **Context Vector**: Weighted sum of the values using the softmax-normalized alignment scores.

### 3. Architecture Details
* **Shared Vocab & Embeddings**: Since both inputs (user queries) and outputs (chatbot responses) are in English, the model uses a shared vocabulary and embedding layer to accelerate semantic learning.
* **Bidirectional Encoder**: Processes user input in both forward and backward directions to capture full context. The forward and backward final states are concatenated to initialize the decoder.
* **Masking**: `mask_zero=True` is enabled in the Embedding layer, instructing the LSTMs and Attention layers to ignore padding tokens (`<pad>`).

---

## Setup & Execution

### Running the Chatbot
1. Run the Python script to train the model on a conversational corpus and evaluate chat predictions:
   ```powershell
   python main.py
   ```
2. **Expected Outputs**:
   - **Terminal logs**: Detailed training loss/accuracy metrics and sample conversational outputs.
   - **`chatbot_attention_results.png`**: A dual-panel dashboard. The left panel shows training convergence. The right panel renders a high-fidelity mock chatbot chat bubble window detailing the conversation.
