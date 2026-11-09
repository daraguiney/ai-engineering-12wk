# Retrieval test set: review sheet

For each question, read the quote and the expected answer, then mark **ok / fix / drop** and add a note if needed.
A good question: (1) the quote alone answers it, (2) it is clear without having read the paper, (3) it is not just the quote's wording turned into a question. For **unanswerable** items, the check is that you agree the corpus does not state the answer.

Reviewed by: ____   Date: ____

---

## q01 · single  _(draft s1_q1)_

**Q:** In the attention-based neural machine translation paper (RNNsearch, Bahdanau et al.), how many sentences contribute to each gradient update direction, and roughly how long was each model trained?

- **1409.0473** (Neural Machine Translation by Jointly Learning to Align and ): “Each SGD update direction is computed using a minibatch of 80 sentences. We trained each model for approximately 5 days.”

**Expected:** 80 sentences per update; about 5 days of training per model.

**Verdict:** ok / fix / drop    **Note:** 

---

## q02 · single  _(draft s1_q2)_

**Q:** In the deep residual learning paper, what test error on CIFAR-10 did the extremely deep network with over a thousand layers obtain?

- **1512.03385** (Deep Residual Learning for Image Recognition): “Our method shows no optimization difﬁculty, and this 103-layer network is able to achieve training error <0.1% (Fig. 6, right). Its test error is still fairly good (7.93%, Table 6).”

**Expected:** 7.93% test error (the 1202-layer network).

**Verdict:** ok / fix / drop    **Note:** 

---

## q03 · single  _(draft s1_q3)_

**Q:** In the original word2vec paper (Efficient Estimation of Word Representations in Vector Space), what initial step size was used for SGD in the Table 2 and 4 experiments, and how was it annealed over training?

- **1301.3781** (Efficient Estimation of Word Representations in Vector Space): “We chose starting learning rate 0.025 and decreased it linearly, so that it approaches zero at the end of the last training epoch.”

**Expected:** Start at 0.025, decreased linearly to approach zero by the end of the last epoch (three epochs).

**Verdict:** ok / fix / drop    **Note:** 

---

## q04 · single  _(draft s1_q4)_

**Q:** When training the Transformer from 'Attention Is All You Need', what value was used for softening the one-hot training targets, and how did this affect perplexity versus accuracy and BLEU?

- **1706.03762** (Attention Is All You Need): “During training, we employed label smoothing of value ϵls = 0.1 [36]. This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.”

**Expected:** 0.1; it hurts perplexity (model is more unsure) but improves accuracy and BLEU.

**Verdict:** ok / fix / drop    **Note:** 

---

## q05 · cross  _(draft s1_q5)_

**Q:** How do the second-moment decay rate and the numerical-stability constant used to train the Transformer differ from the values the original Adam paper recommends as good defaults?

- **1706.03762** (Attention Is All You Need): “We used the Adam optimizer [20] with β1 = 0.9, β2 = 0.98 and ϵ = 10−9.”
- **1412.6980** (Adam: A Method for Stochastic Optimization): “Good default settings for the tested machine learning problems are α = 0.001, β1 = 0.9, β2 = 0.999 and ϵ = 10−8.”

**Expected:** Transformer: beta2=0.98, eps=1e-9; Adam defaults: beta2=0.999, eps=1e-8.

**Verdict:** ok / fix / drop    **Note:** 

---

## q06 · unanswerable  _(draft s1_q6)_

**Q:** How many kilowatt-hours of electricity were consumed in training the base Transformer model on WMT 2014 English-to-German?


**Expected:** Not stated in any paper; Transformer paper reports only training time and FLOPs-based cost estimates, not energy.

**Verdict:** ok / fix / drop    **Note:** 

---

## q07 · single  _(draft s2_q1)_

**Q:** During the original BERT pre-training run, how many tokens go through the model in each update, and how many updates are performed in total?

- **1810.04805** (BERT: Pre-training of Deep Bidirectional Transformers for La): “We train with batch size of 256 sequences (256 sequences * 512 tokens = 128,000 tokens/batch) for 1,000,000 steps, which is approximately 40 epochs over the 3.3 billion word corpus.”

**Expected:** 128,000 tokens per batch (256 sequences x 512 tokens); 1,000,000 steps.

**Verdict:** ok / fix / drop    **Note:** 

---

## q08 · single  _(draft s2_q2)_

**Q:** In the main Dense Passage Retriever configuration, what mini-batch size is used and what extra hard negative is added for every question?

- **2004.04906** (Dense Passage Retrieval for Open-Domain Question Answering): “trained using the in-batch negative setting (Section 3.2) with a batch size of 128 and one additional BM25 negative passage per question.”

**Expected:** Batch size 128, plus one BM25-retrieved negative passage per question.

**Verdict:** ok / fix / drop    **Note:** 

---

## q09 · single  _(draft s2_q3)_

**Q:** What hardware, number of optimisation steps, mini-batch size and learning rate did the REALM authors use for its pre-training phase?

- **2002.08909** (REALM: Retrieval-Augmented Language Model Pre-Training): “We pre-train for 200k steps on 64 Google Cloud TPUs, with a batch size of 512 and a learning rate of 3e-5, using BERT’s default optimizer.”

**Expected:** 64 Google Cloud TPUs, 200k steps, batch size 512, learning rate 3e-5.

**Verdict:** ok / fix / drop    **Note:** 

---

## q10 · single  _(draft s2_q4)_

**Q:** Per the neural language model scaling-law study, if the parameter count is increased, roughly how should the amount of training text grow to keep overfitting in check, and with what exponent?

- **2001.08361** (Scaling Laws for Neural Language Models): “Equation (1.1) and (1.2) together suggest that as we increase the model size, we should increase the dataset size sublinearly according to D ∝N αN αD ∼N 0.74.”

**Expected:** Data should grow sublinearly, D ~ N^0.74.

**Verdict:** ok / fix / drop    **Note:** 

---

## q11 · cross  _(draft s2_q5)_

**Q:** When training the retrieval-augmented components, how does REALM's handling of the document embeddings and search index compare with that of the Retrieval-Augmented Generation (RAG) model of Lewis et al.?

- **2002.08909** (REALM: Retrieval-Augmented Language Model Pre-Training): “by asynchronously re-embedding and re-indexing all documents every several hundred training steps.”
- **2005.11401** (Retrieval-Augmented Generation for Knowledge-Intensive NLP T): “Updating the document encoder BERTd during training is costly as it requires the document index to be periodically updated as REALM does during pre-training”

**Expected:** REALM periodically refreshes (re-embeds/re-indexes) the index during pre-training; RAG keeps document encoder and index fixed because updating is costly.

**Verdict:** ok / fix / drop    **Note:** 

---

## q12 · single  _(draft s3_q1)_

**Q:** In the denoising diffusion model that reached an FID of 3.17 on CIFAR10, how are the noise levels of the forward process set across the sampling chain, and what are the endpoint values?

- **2006.11239** (Denoising Diffusion Probabilistic Models): “We set the forward process variances to constants increasing linearly from β1 = 10−4 to βT = 0.02. These constants were chosen to be small relative to data scaled to [−1, 1]”

**Expected:** Fixed constants rising linearly from 1e-4 to 0.02 (T=1000).

**Verdict:** ok / fix / drop    **Note:** 

---

## q13 · single  _(draft s3_q2)_

**Q:** How much does low-rank adaptation of GPT-3 175B shrink the amount of GPU memory needed during training compared with full fine-tuning?

- **2106.09685** (LoRA: Low-Rank Adaptation of Large Language Models): “On GPT-3 175B, we reduce the VRAM consumption during training from 1.2TB to 350GB.”

**Expected:** From 1.2TB to 350GB.

**Verdict:** ok / fix / drop    **Note:** 

---

## q14 · single  _(draft s3_q3)_

**Q:** For the Retro language model, how large is the document store searched at evaluation time, and how does it differ from the store used during training?

- **2112.04426** (Improving language models by retrieving from trillions of to): “During evaluation the retrieval database consists in the full union of these datasets, with the exception of books for which we use a sub-sample of 4%. The evaluation retrieval database thus contains 1.75T tokens.”

**Expected:** 1.75T tokens: full union of datasets with books subsampled to 4%, versus 600B tokens at training.

**Verdict:** ok / fix / drop    **Note:** 

---

## q15 · single  _(draft s3_q4)_

**Q:** On the 57-subject multitask test introduced by Hendrycks et al., what score do non-specialist crowd workers get?

- **2009.03300** (Measuring Massive Multitask Language Understanding): “Unspecialized humans from Amazon Mechanical Turk obtain 34.5% accuracy on this test.”

**Expected:** 34.5%.

**Verdict:** ok / fix / drop    **Note:** 

---

## q16 · unanswerable  _(draft s3_q5)_

**Q:** What FID score does the DDPM model obtain when trained on 256x256 ImageNet?


**Expected:** Not stated; DDPM reports CIFAR10, CelebA-HQ and LSUN only.

**Verdict:** ok / fix / drop    **Note:** 

---

## q17 · single  _(draft s4_q1)_

**Q:** In the WebGPT browser-assisted question-answering work, how does the approach of drawing several candidate answers and keeping the reward model's top pick differ from reinforcement learning in terms of what it costs?

- **2112.09332** (WebGPT: Browser-assisted question-answering with human feedb): “We used this as an alternative method of optimizing against the reward model, which requires no additional training, but instead uses more inference-time compute.”

**Expected:** Rejection sampling (best-of-n) needs no additional training but uses more inference-time compute.

**Verdict:** ok / fix / drop    **Note:** 

---

## q18 · single  _(draft s4_q2)_

**Q:** What accuracy did the 'Let's think step by step' trigger-phrase method reach on MultiArith with large-scale InstructGPT, compared with the plain zero-shot baseline?

- **2205.11916** (Large Language Models are Zero-Shot Reasoners): “Zero-shot-CoT achieves enormous score gains compared to the zero-shot baseline, e.g. from 17.7% to 78.7% on MultiArith and from 10.4% to 40.7% on GSM8K”

**Expected:** 17.7% (zero-shot) to 78.7% (Zero-shot-CoT) on MultiArith.

**Verdict:** ok / fix / drop    **Note:** 

---

## q19 · single  _(draft s4_q3)_

**Q:** Which optimizer did the Chinchilla compute-optimal language model use instead of the one used for Gopher, and why did the authors prefer it?

- **2203.15556** (Training Compute-Optimal Large Language Models): “We use AdamW (Loshchilov and Hutter, 2019) for Chinchilla rather than Adam (Kingma and Ba, 2014) as this improves the language modelling loss and the downstream task performance”

**Expected:** AdamW instead of Adam; it improves LM loss and downstream performance after finetuning.

**Verdict:** ok / fix / drop    **Note:** 

---

## q20 · cross  _(draft s4_q5)_

**Q:** How does the way answers are decoded from the language model in the original chain-of-thought prompting evaluation differ from the decoding setup in the self-consistency method for LaMDA-137B?

- **2201.11903** (Chain-of-Thought Prompting Elicits Reasoning in Large Langua): “We sample from the models via greedy decoding (though follow-up work shows chain-of-thought prompting can be improved by taking the majority ﬁnal answer over many sampled generations”
- **2203.11171** (Self-Consistency Improves Chain of Thought Reasoning in Lang): “for UL2-20B and LaMDA-137B we applied temperature sampling with T = 0.5 and truncated at the top-k (k = 40) tokens with the highest probability”

**Expected:** CoT paper: greedy decoding; self-consistency: temperature sampling T=0.5, top-k 40, then aggregate.

**Verdict:** ok / fix / drop    **Note:** 

---

## q21 · unanswerable  _(draft s4_q6)_

**Q:** What was the total dollar cost of the OpenAI API calls used to run the Zero-shot-CoT experiments on GSM8K?


**Expected:** Not stated in any paper; expected answer is that the corpus does not say.

**Verdict:** ok / fix / drop    **Note:** 

---

## q22 · single  _(draft s5_q1)_

**Q:** In FlashAttention's gradient computation, why does re-deriving the attention matrix instead of storing it not make training slower, even though it adds arithmetic?

- **2205.14135** (FlashAttention: Fast and Memory-Efficient Exact Attention wi): “In contrast, even with more FLOPs, our recomputation speeds up the backward pass due to reduced HBM accesses”

**Expected:** Because it reduces high-bandwidth-memory accesses, which outweighs the extra FLOPs.

**Verdict:** ok / fix / drop    **Note:** 

---

## q23 · single  _(draft s5_q3)_

**Q:** At roughly what model size did Toolformer's gains from calling external APIs begin to show up in the GPT-2 family experiments?

- **2302.04761** (Toolformer: Language Models Can Teach Themselves to Use Tool): “the ability to leverage the provided tools only emerges at around 775M parameters: smaller models achieve similar performance both with and without tools.”

**Expected:** Around 775M parameters; smaller models perform similarly with or without tools.

**Verdict:** ok / fix / drop    **Note:** 

---

## q24 · single  _(draft s5_q4)_

**Q:** In the manual error analysis of HotpotQA trajectories, how often did chain-of-thought prompting produce a wrong-but-counted-correct answer due to fabricated facts, compared with the reasoning-plus-acting approach?

- **2210.03629** (ReAct: Synergizing Reasoning and Acting in Language Models): “Hallucination is a serious problem for CoT, resulting in much higher false positive rate than ReAct (14% vs. 6%) in success mode”

**Expected:** CoT 14% vs ReAct 6% false positive rate in success mode.

**Verdict:** ok / fix / drop    **Note:** 

---

## q25 · cross  _(draft s5_q5)_

**Q:** Self-Instruct's instruction set generated from GPT-3 and Constitutional AI's red-team prompt set both rely on model-written prompts. Which contains more model-generated prompts, and what are the two counts?

- **2212.10560** (Self-Instruct: Aligning Language Models with Self-Generated ): “We generate a total of over 52K instruc- tions and more than 82K instances corresponding to these instructions after filtering.”
- **2212.08073** (Constitutional AI: Harmlessness from AI Feedback): “generated a further 140,335 prompts by few-shot prompting a pre- trained model, giving a total of 182,831.”

**Expected:** Constitutional AI: 140,335 model-generated red-team prompts vs about 52K (52,445) Self-Instruct instructions; CAI has more.

**Verdict:** ok / fix / drop    **Note:** 

---

## q26 · unanswerable  _(draft s5_q6)_

**Q:** What speedup does FlashAttention achieve when run on Google TPU hardware?


**Expected:** Not stated in any paper; FlashAttention is evaluated on NVIDIA GPUs.

**Verdict:** ok / fix / drop    **Note:** 

---

## q27 · single  _(draft s6_q1)_

**Q:** In LLaMA pretraining, to what fraction of its peak value does the learning rate decay by the end of the cosine schedule, and what weight decay is applied?

- **2302.13971** (LLaMA: Open and Efficient Foundation Language Models): “We use a cosine learning rate schedule, such that the ﬁnal learning rate is equal to 10% of the maxi- mal learning rate. We use a weight decay of 0.1 and gradient clipping of 1.0.”

**Expected:** Final LR is 10% of max; weight decay 0.1.

**Verdict:** ok / fix / drop    **Note:** 

---

## q28 · single  _(draft s6_q2)_

**Q:** In the Generative Agents architecture, what numeric cutoff on accumulated event significance causes an agent to produce higher-level abstract thoughts, and about how often does this happen per simulated day?

- **2304.03442** (Generative Agents: Interactive Simulacra of Human Behavior): “we generate reflections when the sum of the importance scores for the latest events perceived by the agents exceeds a threshold (150 in our implementation). In practice, our agents reflected roughly two or three times a day.”

**Expected:** Threshold 150; roughly two or three times a day.

**Verdict:** ok / fix / drop    **Note:** 

---

## q29 · single  _(draft s6_q3)_

**Q:** For the Game of 24 task, how much does Tree of Thoughts improve when its search keeps five candidate states per step instead of one?

- **2305.10601** (Tree of Thoughts: Deliberate Problem Solving with Large Lang): “ToT with a breadth of b = 1 already achieves a success rate of 45%, while b = 5 achieves 74%.”

**Expected:** Rises from 45% (b=1) to 74% (b=5).

**Verdict:** ok / fix / drop    **Note:** 

---

## q30 · single  _(draft s6_q4)_

**Q:** In QLoRA, what block sizes are used when quantizing the weights and when quantizing the first-level scaling factors, and why is the latter larger?

- **2305.14314** (QLoRA: Efficient Finetuning of Quantized LLMs): “We use a blocksize of 64 for W for higher quantization precision and a blocksize of 256 for c2 to conserve memory.”

**Expected:** 64 for weights (precision), 256 for second-level constants (save memory).

**Verdict:** ok / fix / drop    **Note:** 

---

## q31 · unanswerable  _(draft s6_q5)_

**Q:** How much GPU memory in gigabytes does Tree of Thoughts require when solving Mini Crosswords with GPT-4?


**Expected:** Not stated; ToT uses API calls to GPT-4 and reports token cost, not GPU memory.

**Verdict:** ok / fix / drop    **Note:** 

---

## q32 · single  _(draft s7_q2)_

**Q:** Why did the vLLM authors settle on a particular number of tokens per KV cache block as the default, rather than using much larger blocks?

- **2309.06180** (Efficient Memory Management for Large Language Model Serving): “we find that the block size 16 is large enough to efficiently utilize the GPU and small enough to avoid significant internal fragmentation in most workloads.”

**Expected:** Block size 16: big enough to use the GPU efficiently, small enough to avoid significant internal fragmentation.

**Verdict:** ok / fix / drop    **Note:** 

---

## q33 · single  _(draft s7_q3)_

**Q:** In the Minecraft lifelong-learning agent Voyager, how much did the count of discovered items fall when the automatic curriculum was swapped for a random one?

- **2305.16291** (Voyager: An Open-Ended Embodied Agent with Large Language Mo): “The discovered item count drops by 93% if the curriculum is replaced with a random one, because certain tasks may be too challenging if attempted out of order.”

**Expected:** 93% drop.

**Verdict:** ok / fix / drop    **Note:** 

---

## q34 · single  _(draft s7_q4)_

**Q:** For the 'Lost in the Middle' study of long-context language models, how much can multi-document QA accuracy of GPT-3.5-Turbo fall when the relevant document sits mid-context?

- **2307.03172** (Lost in the Middle: How Language Models Use Long Contexts): “GPT-3.5-Turbo’s multi-document QA performance can drop by more than 20%—in the worst case, performance in 20- and 30-document settings is lower than per-formance without any input documents”

**Expected:** By more than 20%; in worst case below closed-book performance.

**Verdict:** ok / fix / drop    **Note:** 

---

## q35 · cross  _(draft s7_q5)_

**Q:** How well do GPT-4 judgments line up with human annotators, as reported when the papers on LLM-as-a-judge (MT-Bench) and on Direct Preference Optimization justify using GPT-4 as an evaluator?

- **2306.05685** (Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena): “The agreement under setup S2 (w/o tie) between GPT-4 and humans reaches 85%, which is even higher than the agreement among humans (81%).”
- **2305.18290** (Direct Preference Optimization: Your Language Model is Secre): “We find GPT-4 judgments correlate strongly with humans, with human agreement with GPT-4 typically similar or higher than inter-human annotator agreement.”

**Expected:** MT-Bench: 85% GPT-4-human vs 81% human-human (non-tie). DPO: GPT-4 agreement with humans similar or higher than inter-annotator agreement.

**Verdict:** ok / fix / drop    **Note:** 

---

## q36 · unanswerable  _(draft s7_q6)_

**Q:** What DPO beta value and learning rate were used when fine-tuning Llama-2 70B on the Anthropic HH dialogue dataset?


**Expected:** Not stated in corpus; DPO paper predates Llama-2 experiments.

**Verdict:** ok / fix / drop    **Note:** 

---

## q37 · single  _(draft s8_q1)_

**Q:** In Mistral 7B, how much does the fixed-size rolling cache cut memory use when processing a 32k-token sequence, and what position in the cache holds the entry for timestep i?

- **2310.06825** (Mistral 7B): “the keys and values for the timestep i are stored in position i mod W of the cache. As a result, when the position i is larger than W, past values in the cache are overwritten”

**Expected:** Position i mod W (the 8x memory reduction at 32k is in the adjacent sentence).

**Verdict:** ok / fix / drop    **Note:** 

---

## q38 · single  _(draft s8_q2)_

**Q:** What value does Self-RAG use for its decision threshold on whether to fetch passages for most tasks, and why is it set differently for ALCE?

- **2310.11511** (Self-RAG: Learning to Retrieve, Generate, and Critique throu): “To encourage frequent retrieval, we set the retrieval threshold to 0.2 for most tasks and to 0 for ALCE (Gao et al., 2023) due to citation requirements.”

**Expected:** 0.2 for most tasks; 0 for ALCE because of citation requirements.

**Verdict:** ok / fix / drop    **Note:** 

---

## q39 · single  _(draft s8_q3)_

**Q:** On the synthetic task where the gaps between relevant tokens are randomized, how far did the Mamba selective SSM generalize relative to its training length, and how far did competing methods get?

- **2312.00752** (Mamba: Linear-Time Sequence Modeling with Selective State Sp): “It generalizes perfectly to million-length sequences, or 4000× longer than it saw during training, while no other method goes beyond 2×.”

**Expected:** Mamba: 4000x training length (1M tokens); others no beyond 2x.

**Verdict:** ok / fix / drop    **Note:** 

---

## q40 · single  _(draft s8_q4)_

**Q:** According to the retrieval-augmented generation survey, what problems arise in the retrieval step of the earliest RAG paradigm?

- **2312.10997** (Retrieval-Augmented Generation for Large Language Models: A ): “The retrieval phase often struggles with precision and recall, leading to the selection of misaligned or irrelevant chunks, and the missing of crucial information.”

**Expected:** Struggles with precision and recall: misaligned/irrelevant chunks selected, crucial information missed.

**Verdict:** ok / fix / drop    **Note:** 

---
