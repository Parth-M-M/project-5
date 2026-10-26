# Component B — Recorded Video Explanation Guide (5–10 Minutes, Face PiP + Screen)

**File Naming Convention:** `DL_PR5_YourName_GRID.mp4`  
**Format Requirements:** MP4 or MOV (max 500 MB), 5–10 minutes, webcam picture-in-picture (PiP) visible throughout, screen showing `DL_PR5.ipynb`.

---

## Time-Stamped Speaking Script Covering All 8 Required Concepts (Criterion B2 & B3)

### 1. (0:00 – 1:00) Why Word Order Matters in Text & Why the ANN Baseline Fails to Use It (Task 1 & Task 2.1)
- **What to show on screen:** Scroll through Task 1 (`length_distribution.png`) and Task 2.1 (`ANN_64` model summary).
- **What to say:**
  > "In Task 1, we loaded the 50,000 raw IMDB movie reviews, removed 418 exact duplicates *before* splitting to prevent data leakage, cleaned HTML tags while keeping apostrophes and negation stop-words like `'not'`, and pre-padded sequences to `MAXLEN = 300` so the final time step read by an RNN always sees real words instead of trailing zeros.  
  > Movie reviews are strictly sequential: `'the movie was not good'` and `'the movie was good, not bad'` contain the same words with opposite meanings. In Task 2.1, our order-blind ANN baseline flattens `(300, 32)` into `9,600` independent slots. Because it cannot share weights across positions, its `Dense(64)` layer explodes to `614,464` parameters (`934,529` total), memorising the training set while ignoring sequential word order."

### 2. (1:00 – 2:10) The RNN Equation, Hidden State & Weight Sharing via Manual NumPy Forward Pass (Task 2.2 – 2.5)
- **What to show on screen:** Scroll to Task 2.3 & 2.4 (`rnn_forward_heatmap.png` and `np.allclose = True` output).
- **What to say:**
  > "An RNN solves this by reading one word $x(t)$ at a time and updating a running hidden state $h(t) = \tanh(W_{xh} x(t) + W_{hh} h(t-1) + b)$. Because the exact same matrices $W_{xh}$, $W_{hh}$, and $b$ are shared at every time step, `SimpleRNN(64)` needs only $64 \times (32 + 64 + 1) = 6,208$ parameters regardless of sequence length!  
  > In Task 2.3, I implemented a 4-step RNN cell from scratch in NumPy (`hidden_size=3, input_size=2`), plotted the hidden states in `rnn_forward_heatmap.png`, and copied the transposed weights (`Wxh.T, Whh.T, b`) into Keras `SimpleRNN`, confirming an exact match with `np.allclose = True`."

### 3. (2:10 – 3:00) The Four Types of RNN & Which One Sentiment Analysis Uses (Task 3.1 – 3.5)
- **What to show on screen:** Scroll to Task 3.1 shape outputs and Task 3.4 `simplernn_training_curves.png`.
- **What to say:**
  > "In Task 3.1, we demonstrated all four RNN topologies on dummy tensors: One-to-One `(4, 1)` for tabular data, One-to-Many `(4, 10, 16)` for image captioning, Many-to-One `(4, 16)` with `return_sequences=False`, and Many-to-Many `(4, 10, 16)` with `return_sequences=True` for NER or translation.  
  > Sentiment analysis uses **Many-to-One** because an entire 300-word movie review maps to a single binary sentiment label. When we trained `SimpleRNN(64)`, it learned slowly and plateaued at **80.42% test accuracy** (`0.8768` ROC-AUC)."

### 4. (3:00 – 4:10) Backpropagation Through Time (BPTT) & Why the Gradient Is a Product of Many Terms (Task 4.1 – 4.3)
- **What to show on screen:** Scroll to Task 4.1 equations and Task 4.3 `gradient_norm_simplernn.png`.
- **What to say:**
  > "Why did `SimpleRNN` struggle on 300-word reviews? During Backpropagation Through Time (BPTT), the RNN is unrolled across $T=300$ steps. By the chain rule, the gradient of the loss at step $T$ with respect to the earliest hidden state $h(1)$ is a **product of 299 Jacobian matrices**: $\partial L/\partial h(1) = \partial L/\partial h(T) \prod W_{hh}^T \text{diag}(1 - h(t)^2)$.  
  > Using `tf.GradientTape` in Task 4.3, we measured the gradient norm at every time step and saw it collapse from `8.94e-2` at $t=300$ down to `1.67e-9` at $t=1$—a 53-million-fold collapse!"

### 5. (4:10 – 5:00) Vanishing vs. Exploding Gradients & Why Clipping Fixes Only One (Task 4.4 – 4.6)
- **What to show on screen:** Show `simplernn_acc_vs_seqlen.png` and `exploding_gradient_clipping.png`.
- **What to say:**
  > "Because $\tanh'$ is at most 1 and usually much smaller, multiplying factors below 1 hundreds of times causes **vanishing gradients**, giving `SimpleRNN` an effective memory of only 10–20 words—which is why accuracy actually dropped from `83.42%` at length 100 down to `80.42%` at length 300.  
  > Conversely, when factors exceed 1 (such as with a high learning rate `lr=0.05` in Task 4.5), the product grows exponentially—**exploding gradients**, where loss spiked to `28.41`. Adding `clipnorm=1.0` rescales the gradient vector whenever its norm exceeds 1.0, completely stabilising the loss (`0.5045`). However, clipping only shrinks huge gradients—it cannot amplify vanished gradients from early time steps."

### 6. (5:00 – 6:15) LSTM Gates & The Cell State as a Protected Memory Path (Task 5.1 – 5.7)
- **What to show on screen:** Show Task 5.1 equations, Task 5.4 NumPy table, and Task 5.6 `gradient_norm_lstm_vs_rnn.png`.
- **What to say:**
  > "To fix vanishing gradients, Task 5 introduces the **LSTM**, which adds a dedicated **cell state conveyor belt $C(t) = f(t) \odot C(t-1) + i(t) \odot g(t)$** regulated by the forget gate $f(t)$, input gate $i(t)$, candidate $g(t)$, and output gate $o(t)$. Because $C(t)$ is updated by **addition**, the gradient along $C(t)$ is multiplied by $f(t) \approx 1$ rather than $W_{hh}$ and $\tanh'$!  
  > With 4 gates, `LSTM(64)` has $4 \times 6,208 = 24,832$ recurrent parameters. In our NumPy table at Step 2 Unit 2, when $f \approx 0.88$ and $i \approx 0.04$, $C(t)$ stays preserved at `0.2031`. On IMDB, `LSTM(64)` jumps to **88.64% test accuracy** (`0.9542` AUC), and its gradient at $t=1$ is **1.93 million times larger** than `SimpleRNN`!"

### 7. (6:15 – 7:15) GRU Update and Reset Gates & How GRU Differs from LSTM (Task 6.1 – 6.6)
- **What to show on screen:** Show Task 6.2 GRU vs LSTM table, `training_curves_rnn_lstm_gru.png`, and `gradient_norms_all_cells.png`.
- **What to say:**
  > "In Task 6, we built the **GRU**, which removes the separate cell state and merges the forget and input gates into a single **update gate $z(t)$** alongside a **reset gate $r(t)$**: $h(t) = (1 - z(t)) \odot h(t-1) + z(t) \odot \tilde{h}(t)$.  
  > With 3 gates and `reset_after=True`, `GRU(64)` has `18,816` recurrent parameters (~75% of LSTM), trains **19% faster** (`15.1 s/epoch` vs `18.6 s/epoch`), and achieves **88.48% test accuracy** (`0.9531` AUC). Our hidden-unit sensitivity study (`32, 64, 128`) confirmed `units=64` as the point of diminishing returns."

### 8. (7:15 – 8:30) Final Recommendation, Gradient-Norm Proof & Live `predict_sentiment()` Demo (Task 7.1 – 7.6)
- **What to show on screen:** Show `results_table.png`, `roc_curves.png`, `accuracy_by_length.png`, and run the `predict_sentiment()` cell live in Task 7.5!
- **What to say:**
  > "In Task 7, our 5-model comparison table and `accuracy_by_length.png` chart show that on long reviews (>250 words), `SimpleRNN` drops to `74.1%` whereas `LSTM` (`87.35%`) and `GRU` (`87.10%`) stay strong—directly proving what our gradient-norm plot showed. For a production service, we recommend **single-layer `GRU(64)`** because it matches `LSTM` within `0.16%` accuracy while cutting recurrent parameters by 24% and training/inference time by 19%.  
  > Finally, let's run `predict_sentiment()` live: it correctly classifies positive, negative, single negation (`'not good'` -> negative `0.1425`), and double negation (`'not bad at all'` -> positive `0.7864`), while illustrating why heavy sarcasm (`'masterpiece of boredom'`) still fools unidirectional models trained without pretrained embeddings."
