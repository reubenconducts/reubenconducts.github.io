---
title: I think you just fell out of a Coconut tree!
description: Exploring inference-time scaling and optimization in language models
tags: [LLM, Inference]
date: 2025-04-01
---

# {{ $frontmatter.title }}


Language models (LM) have exhibited impressive performance on a wide variety of tasks, from language modeling to translation. Very large language models have been shown to solve basic reasoning tasks, like arithmetic, coding, and state tracking. Smaller models, however, continue to struggle with even very basic reasoning: Gemma 3-4B, a 4 billion parameter model released this March, reaches only 38.4% accuracy on grade-school arithmetic tasks ([Gemma 3 technical report](https://arxiv.org/abs/2503.19786v1)).

In the past few years, *inference-time scaling* has emerged as the predominant paradigm for improving LM reasoning capabilities. The more tokens we allow LMs to generate *during* inference, the deeper reasoning problems they are able to solve. This approach underlies OpenAI's recent o-series and Deepseek's r1 models, which leverage specific and additional inference-time compute to achieve state-of-the-art results on reasoning tasks. Scaling inference-time compute can be quite computationally expensive, though; this is a potential bottleneck as LMs are increasingly used in real-world applications where inference time is critical or compute and memory resources are limited. Thus, two orthogonal optimization goals have emerged: increasing the accuracy of the model on complex reasoning tasks and decreasing the computational cost of inference.

In this blog post, we will discuss several means for decreasing the computational cost of inference in reasoning tasks, which broadly fall into two categories:

1. **Decreasing the size of the model**
   - Quantizing the model (using low-floating-point arithmetic)
   - Using model compression techniques
   - *Pruning* the model (remove parameters)
2. **Decreasing the total number of computations**
   - Encouraging the LM to generate fewer tokens during reasoning
   - Forcing the LM to skip certain steps in its forward pass


In this blog post, we will address both of these goals by describing joint work-in-progress with Belinda Li on 1) structured pruning of LMs and 2) *continuous chain-of-thought* prompting.


# *Aide-mémoire* on Language Models


While we assume basic familiarity with the transformer architecture in this blog post, we provide a brief overview for convenience and to set notation. 

> **Definition 1**
>
> A **language model** is a statistical model that takes in a sequence of text (split up into *tokens*) and outputs a probability distribution over the vocabulary of tokens:
>
> $$
> p(x_1, x_2, \dots, x_{t-1})
> $$

Modern-day language models are typically implemented through the Transformer architecture ([Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762)), and we will focus on these in this post.

> **Definition 2** 
> A **(decoder-only) Transformer language model** (or **LM** for brevity) is a neural network language model consisting of three sections:
>1. An **embedding layer** that maps tokens to vectors in some fixed-dimensional vector space, of dimension far smaller than the vocabulary size.
>2. A stack of **attention layers**, each consisting of a handful of attention heads and a feed-forward layer.
>3. A **projection layer** that maps the output of the final attention layer to a vector in the vocabulary space.


We will write $\vec{x} = (x_1, x_2, \dots, x_t)$ to denote a sequence of tokens. To generate the next token, models may employ a handful of decision techniques, the most common being **top-$p$-sampling**, where the model computes $p(x_{t+1} \vert \vec{x})$ for all $x_{t+1}$ in the vocabulary, chooses the smallest set of tokens whose cumulative probability is greater than some threshhold $p$, and then samples one token from this top-$p$ set (renormalizing their probabilities to sum to 1). This allows for a small amount of randomness in token generation, leading to models that do not get stuck in circles of echolalia. 

For the sake of concision, we will write $\mathsf{N}^{\mathcal{L}}_i(\vec{x})$ to denote the $i$-th next-token prediction of the LM $\mathcal{L}$ for the sequence $\vec{x}$ (here, $\mathsf{N}$ stands for "next").

This is defined recursively: given the <span>$i$</span>-th next-token prediction 
<span>$\mathsf{N}^{\mathcal{L}}_i(\vec{x})$</span>, 
the <span>$(i+1)$</span>-th next-token prediction <span>$\mathsf{N}^{\mathcal{L}}_{i+1}(\vec{x})$</span> is defined as

$$
\mathsf{N}^{\mathcal{L}}_{i+1}(\vec{x}) = \mathsf{N}^{\mathcal{L}}_1\left(\vec{x}, \mathsf{N}^{\mathcal{L}}_1(\vec{x}), \dots, \mathsf{N}^{\mathcal{L}}_i(\vec{x})\right)
$$

> **Aside**
>
> In this post, we will only be considering *autoregressive* LMs, those trained on next-token prediction, and not models trained on masked language modeling (like BERT).

The **embedding** layer in an LM takes tokens from vocabulary space (high-dimensional but exceedingly sparse) to a "latent space" with some *hidden dimension* $d$ via an embedding that is learned to turn semantically-meaningful relationships into geometrically-meaningful ones (see [Mikolov et al. (2013)](https://arxiv.org/pdf/1301.3781) for early work and 3Blue1Brown's [excellent video (12:27)](https://www.youtube.com/watch?v=wjZofJX0v4M&t=1457s) for a visualization of this idea), while the **projection** layer does the opposite, taking vectors in latent space to vectors in vocabulary space. If generating a token, this unembedded vector will be passed to top-$p$-sampling (or some other decision technique) to output a token. 

The embedding and projection layers of an LM are learned independently from the rest of the model (and in particular may be shared by families of models), so the rest is what we focus on. 

## Attention


**Attention**, specifically self-attention, is the backbone of LMs, allowing them to learn dependencies between any two tokens in a sequence. For a lucid exposition of the attention mechanism, see [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/). For the purposes of this blog post, we will only need the following:


> **Definition 3**
>
> An **attention head** consists of a *query* matrix $W^{(q)}$, a *key* matrix $W^{(k)}$, and a *value* matrix $W^{(v)}$, each with dimension $d \times d$, where $d$ is the embedding dimension of the model. The forward pass is computed as follows:
> - Given an input matrix $X$ of dimension $n \times d$ (corresponding to a sequence of $n$ tokens), the attention head computes matrices
>   - $Q = XW^{(q)}$
>   - $K = XW^{(k)}$
>   - $V = XW^{(v)}$
>   of dimension $n \times d$.
> - It then computes the **scaled dot-product attention** of $Q$, $K$, and $V$ as
> $$
> \textsf{Attention}(Q, K, V) = \textsf{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V~.
> $$
> A model may contain multiple attention heads $h_1, \dots, h_H$, computed in parallel, the outputs of which are then concatenated to form a single *multi-head attention* tensor $\textsf{MHA}_{h_1, \dots, h_H}(X)$.

The Transformer combines attention heads with fully-connected feed-forward networks to add in nonlinearity needed for the model to posess flexibility.

> **Definition 4**
>
> An **attention layer** (or *transformer block*) consists of
> - A stack of attention heads $h_1, \dots, h_H$ for some integer $H$, combining to give a multi-head attention block $\textsf{MHA}(X) = \textsf{MHA}_{h_1, \dots, h_H}(X)$
> - A feed-forward layer $\textsf{MLP}(Y)$, often taken to have one hidden layer.
>
> The forward pass of the complete attention layer is given by
> $$
> \textsf{MLP}(\textsf{LayerNorm}(X + \textsf{MHA}(\textsf{LayerNorm}(X)))) + X~,
> $$
> where $\textsf{LayerNorm}$ is some form of layer normalization and $X$ is added onto the end of each layer as a *residual stream*.

> **Aside**
>
> The original Transformer paper (Vaswani et al. (2017)) used a *post-layernorm* architecture, where $\mathsf{LayerNorm}$ was applied *after* $\textsf{MHA}$ and $\textsf{MLP}$, rather than before as we have done. Our *pre-layernorm* architecture has become ubiquitous. 


While the majority of the parameters in an LM are in the transformer layer MLPs, as sequence length grows (these days, LLMs may accept sequences with upwards of 2 million tokens), the attention heads become the most computationally intensive component of the model. 

### Computational Complexity of an Attention Layer

Let's do a quick calculation to determine how many FLOPs are used in the forward pass of a given attention layer. Fixing notation, let's assume we have
- embedding dimension $d$, sometimes called *hidden size*
- number of attention heads $H$
- per-head dimension $d_h = d / H$
- MLP intermediate layer size $d_\textsf{MLP}$, taken to be $4d$ as is common

To start, let's consider multi-head attention. Given our input sequence, a tensor $X$ of dimension $[N, d]$, for each attention head $h_i$, we perform for the multiplications $Q_i = XW_i^{(q)}, K_i = XW_i^{(k)}$, and $V_i = XW_i^{(v)}$, where each of $W_i^{(q, k, v)}$ has dimension $[d, d_h]$. This gives

$$
3 \cdot (2 \cdot N \cdot d \cdot d_h) \cdot H = 6 \cdot N \cdot d^2
$$

FLOPs, counting multiply-adds as 2 FLOPs. Next, for each head we compute the attention score $\mathsf{Softmax}(Q_iK_i^\top)V$. The inner matrix multiplication takes $2 \cdot N^2 \cdot d_h$ FLOPs, while the softmax takes roughly $5N^2$ FLOPs (for each of the $N$ rows, roughly $N$ from the $\mathsf{max}$ computation, $N$ subtractions, $N$ exponentiations, $N$ additions, and $N$ divisions). The outer matrix multiplication takes another $2N^2d_h$ FLOPs, giving a total of 

$$
6Nd^2 + 4N^2d + 5N^2H
$$

FLOPs in the multi-head attention sub-layer. 

Moving onto the MLP, we have one layer of shape $[d, 4d]$ and another of shape $[4d, d]$. Passing the output of attention through, we have $2 \cdot N \cdot d \cdot 4d$ FLOPs per layer, giving a total of

$$
16Nd^2
$$

FLOPs in the MLP. 

Because layer norm and adding the residual stream are negligible computationally in larger models, we will ignore them. In the end, the approximate FLOPs of a single attention layer is

$$
24Nd^2 + 4N^2d + 5N^2H = 24Nd^2 + N^2(4d + 5H)~.
$$

Taking a common $d = 512$ and $H = 8$, the $N^2$ term dominates at sequences of length >2900, which are commonplace in reasoning chains. This is to say, attention takes over as the most computationally intensive part of the model with larger sequence lengths (even though the majority of parameters are in the MLPs!).

# Structured Pruning

For years, it has been known that neural networks have extensive redundancy in their weights ([Le Cun et al. (1990)](https://proceedings.neurips.cc/paper_files/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf), [Han et al. (2015)](https://arxiv.org/pdf/1506.02626), [Frankle and Carbin (2019)](https://arxiv.org/pdf/1803.03635)), and that pruning these weights can lead to significant gains in efficiency with little-to-no loss in accuracy. Large neural networks exhibit a phenomenon known as **double descent** ([Belkin et al. 2019](https://arxiv.org/abs/1812.11118) for the original paper and [a lucid exposition](https://arxiv.org/pdf/2303.14151)), where a model appears to overfit in training (as is predicted by statistical learning theory) before beginning to generalize. The intuition behind this unintuitive result is that large models have sub-networks that generalize better than the network as a whole. These sub-networks are highly contingent on the randomization of weights at the beginning of training and emerge only through the training process. That is to say, while the entirety of the network may not be terribly useful during inference, it is critical during training. 

As a result, large swaths of neural networks may be deleted in their entirety (outright, or in a task-specific manner) with essentially no decrease in accuracy. Determining which parts are worth keeping is a challenging task: in an MLP with three hidden layers each 10 neurons, the "minimal subnetwork" may have only 9 neurons total, but determining which these are could take a search of over 14 million possibilities (each of which would need to be evaluated on a sufficiently large validation set). A more efficient yet slightly sub-optimal process for pruning models would be beneficial, and this comes in the form of **structured pruning**. 

## Pruning Attention Heads

As we noted above, attention is the most compute-intensive aspect of LMs as sequence length grows very long. We can save a large amount of computation by removing entire attention heads, directly inputting zeros into the MLP layer where their outputs would be. Suppose we have an LM with 8 attention heads, embedding dimension 512, and MLP with one hidden layer of intermediate dimension 1024. Given an input sequence with 1000 tokens, the attention layer takes 5,169,152,000 FLOPs and the MLP takes 4,194,304,000 FLOPs. If we are able to prune two attention heads in this layer, we will save 

$$
6 \times 1000^2 \times 128 = 768,000,000
$$

FLOPs in the attention sub-layer, about 8% of FLOPs for the layer as a whole. If we are able to prune 6 attention heads, we'd save over 25% of FLOPs for the layer. 

This may not be as drastic or damaging as it sounds: results by [Voita et al. (2019)](https://arxiv.org/pdf/1905.09418) show that upwards of 90% of attention heads in some layers can be pruned with only a little impact on performance. Therein, the authors expand the idea of **layer-wise relevance propagation** (until then used for relevance of specific neurons, see [Bach et al. (2015)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140)) to determine relevance of attention heads for specific tasks. By fine-tuning a transformer with a regularizing objective that encourages attention-head dropout, they were able to cut heads in half across the model as a whole with only 1% loss in accuracy. 

Alternatively, one may use the effiency gains from attention head pruning to create models that are even more accurate for a given parameter count ([Shim et al. (2023)](https://arxiv.org/pdf/2110.03252)).

## Pruning Entire Layers

Many techniques for pruning models are subtle: excising single neurons or even attention heads. What about entire layers? By pruning a layer, we are able to save nearly $1/L$ times the model FLOPs (where $L$ is the number of layers), an enormous amount. In [Gromov et al. (2025)](https://arxiv.org/pdf/2403.17887), the authors find that up to *half* of the layers in a model can be removed before experiencing noticeable accuracy gains, leading to a nearly 100% speedup in inference. The authors use *angular block distance*, a cosine similarity measure of how much a block of layers influences the residual stream, to determine which layers to delete. Interestingly, they determined that different tasks respond differently to layer deletion: performance on the GSM8K grade-school arithmetic dataset degrades almost immediately, while that on the MMLU dataset is far more robust to layer deletion.

Given that layers account for nearly all FLOPs in a model, deleting layers appears to be the easiest and most immediately beneficial form of structured pruning, from a computational cost perspective. Faster inference affords models the chance to generate more tokens within the same time frame, which, as we will see below, greatly expands their reasoning capacity. 

# Chain-of-Thought Prompting


One of the most successful strategies for improving LM reasoning ability has been to prompt the model to reason through tasks step-by-step, with a series of intermediate reasoning steps, utilizing more tokens to make up for any lack in inherent ability of the base model. This makes intuitive sense for models trained on next-token-prediction: by performing small steps bit-by-bit, essentially tracking the state of the task, the model is more likely to maintain an accurate representation of the task at hand. This approach to reasoning is called **chain of thought prompting (CoT)**.

Chain-of-thought prompting was systematically studied in a [landmark paper](https://arxiv.org/pdf/2201.11903) by Wei et al. in 2022. Therein, the authors demonstrate on multiple LMs of multiple sizes that training a model to verbalize its reasoning can lead to significant gains in reasoning ability on certain complex reasoning tasks.


<div style="display: flex; flex-direction: row; flex-wrap: wrap; margin-bottom: 20px; align-items: center;">
  <div style="flex: 1 1 300px; margin-right: 20px; margin-bottom: 15px; font-size: 0.9em; min-width: 250px;">
    <strong>Figure 1:</strong> Summary of results from Wei et al. (2022). They study three different datasets including GSM8K (a dataset of grade-school math word problems) with three groups of models of varying sizes. Their results show that CoT prompting leads to significant reasoning gains on larger models when reasoning about tasks that scale roughly linearly.
  </div>
  <div style="flex: 2 1 400px;">
    <img src="../images/cot_original_summary.png" alt="Summary of CoT results" style="max-width: 90%; height: auto; display: block; margin: 0 auto;">
  </div>
</div>


While these results are indicative of greater possibilities for LMs, they are incomplete: rather than fine-tuning the models to get better at reasoning, the authors merely prompt the model to reason by providing a series of templates for it to base its reasoning on. This perhaps explains why the gains are most significant on larger models, which are better able to make use of the reasoning cues.


Could we perhaps fine-tune models on reasoning chains to better improve their task-specific reasoning ability? [Nye et al. (2021)](https://arxiv.org/abs/2112.00114) began to explore this possibility, by training a model to generate specifically-formatted reasoning chains. They teach models to use a "scratchpad" to show their work, training them step-by-step to verbalize the correct reasoning chains. At a high level, their curriculum trains models to be able to generate correct reasoning chains bit-by-bit: as the epochs progress, the model is tasked with generating a larger proportion of the chain, working backwards.


> **Example Prompt and Reasoning**
>
> Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?
>
> Scratchpad:
> 1. Roger has 5 tennis balls.
> 2. He buys 2 more cans of tennis balls.
> 3. Each can has 3 tennis balls.
> 4. He has 5 + 2 = 7 cans of tennis balls.
> 5. He has 7 cans * 3 tennis balls/can = 21 tennis balls.
>
> A: 21


They begin the curriculum by training the model to generate the answer—21 in this case—given the entirety of the reasoning chain. Stage-by-stage, they delete the final tokens of the reasoning chain and train the model to fill in the remaining tokens and generate the correct answer.


The accuracy gains were significant: with 9-digit addition, they were able to achieve 96% accuracy with a 1B parameter model, compared to near 0% accuracy without the scratchpad.


While their results showcase the impressive accuracy gains possible from training a model to generate better reasoning chains, the computational cost of CoT inference is still high, requiring more extensive token-generation than standard prompting. Recent work has begun to address this issue from two directions: teaching a model to *internalize* its CoT process and foregoing the generation of tokens altogether, feeding the last layer's activations directly back into the first layer of the model, allowing it to reason *continuously* in latent space (this is a concept we will return to later).


## Internalizing Chain-of-Thought


Humans are very good at reasoning *without* explicitly verbalizing our reasoning steps. Some may consider this "intuition", an aspect of intelligence common to nearly all animals, but one that is quite challenging to formalize, and hence quite difficult to teach to LMs. Rather than training a model to output explicit reasoning chains, we could instead imagine training a model to *internalize* its CoT process, removing reasoning steps throughout the training process.


This is precisely what [Deng et al. (2023)](https://arxiv.org/pdf/2311.01460) and [Deng et al. (2024)](https://arxiv.org/pdf/2405.14838v1) investigated in their papers "Implicit Chain-of-Thought Reasoning via Knowledge Distillation" and "From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step". Both papers explore different ways to have models internalize their CoT process, theoretically allowing them to solve tasks that have required CoT without the additional token generation overhead. Via [knowledge distillation](https://arxiv.org/abs/2402.13116), they are able to achieve nearly identical performance to explicit CoT prompting with nearly 5x faster inference.


<div style="display: flex; flex-direction: row; flex-wrap: wrap; margin-bottom: 20px; align-items: center;">
  <div style="flex: 1 1 300px; margin-right: 20px; margin-bottom: 15px; font-size: 0.9em; min-width: 250px;">
    <strong>Figure 2:</strong> Summary of results from Deng et al. (2023). They study GSM8K, 4-digit multiplication, and 5-digit multiplication. They compare reasoning without CoT (No-CoT), with explicit CoT, and with implicit CoT via knowledge distillation.
  </div>
  <div style="flex: 2 1 400px;">
    <img src="../images/implicit_knowledge_distill.png" alt="Summary of implicit CoT results" style="max-width: 90%; height: auto; display: block; margin: 0 auto;">
  </div>
</div>


In the absence of a suitable "teacher" LM for knowledge distillation, it would be helpful to have a different method for training a model to internalize its CoT process. Deng et al. (2024) in their follow-up work implement a curriculum to train models to internalize their thought processes step-by-step, allowing them to interpolate between fully-explicit and fully-implicit CoT. This curriculum achieves better accuracy than knowledge distillation (ICoT-KD), among other benefits. The authors summarize the benefits of stepwise-internalized ICoT as follows:


> Compared to ICoT-KD, Stepwise Internalization has three advantages:
> First, it is simpler to implement as it does not require a teacher model. Second, while ICoT-KD
> internalizes reasoning into a single "column" of states (corresponding to the final input position),
> Stepwise Internalization allows the model to internalize reasoning across all input positions. Lastly,
> Stepwise Internalization achieves better accuracy compared to ICoT-KD.

## Coconut: Chain of Continuous Thought

In December of 2024, researchers at Meta AI released a paper titled [Training Large Langage Models to Reason in a Continuous Latent Space (Hao et al. 2024)](https://arxiv.org/abs/2412.06769). The impetus behind their work is the idea that token generation in language models is an incredibly lossy process: when passing through the unembedding layer, a vector in $\mathbb{R}^d$ (where $d$ is the embedding dimension of the model) is quantized into one of $L$ tokens, where $L$ is the vocabulary size. Moreover, in chain of thought prompting, this can use extraneous compute resources, as the output token must then be re-embedded before passing through the first transformer layer when generating a subsequent token. The core insight in their paper—termed "Chain of Continuous Thought" or "Coconut"—is that a model could potentially gain comparable or better accuracy to "vanilla" CoT with fewer tokens and less compute used per token. 

### High-Level Overview

Coconut is a fine-tuning and test-time paradigm. The authors begin with a pretrained LM

$$
\mathcal{M} = V \xrightarrow{e} \mathbb{R}^d \xrightarrow{\mathsf{Transformer}} \mathbb{R}^d \xrightarrow{u} V~,
$$ 

where $V$ is the set of tokens in the vocabulary and $d$ is the embedding dimension (in their paper, they take GPT-2). Step-by-step, they train $\mathcal{M}$ to use more tokens during a "latent phase" where transformer layer outputs are directly fed back into the first transformer layer, before jumping into "language mode" and generating tokens normally. They use cross-entropy loss on only the normally-generated tokens to allow the model to learn its own internal methods for solving various problems. 

To be precise, given a sequence $\vec{x} = (x_1, \dots, x_n)$ of input tokens and allowing the model $\ell$ latent thoughts, the model autoregressively computes latent thoughts 

$$
t_i = \mathsf{Transformer}(e(x_1), \dots, e(x_n), t_1, \dots, t_{i-1})
$$

before allowing the normally-generated tokens $w_j$ to be generated recursively as

$$
w_j = u\left(\mathsf{Transformer}\left(\overbrace{e(x_1), \dots, e(x_n)}^{\text{input tokens}}, \overbrace{t_1, \dots, t_{\ell}}^{\text{latent thoughts}}, \overbrace{e(w_1), \dots, e(w_{j-1})}^{\text{normal tokens}}\right)\right)
$$

### Results

Coconut surpasses vanilla CoT on a handful of tasks ([ProntoQA](https://arxiv.org/abs/2210.01240) and ProsQA, introduced in the Coconut paper) by a wide margin, with significant decreases in token generation (ten times fewer in the case of ProntoQA), while it comes somewhat close to vanilla CoT on the [GSM8K dataset](https://arxiv.org/abs/2110.14168), with three times fewer tokens generated. These suggest that models employing Coconut are effectively learning to utilize the additional flexibility afforded to them by reasoning in latent space.

### Coconut Curriculum

Given that LMs have not generally been trained to reason continuously in latent space, the exact training curriculum is crucial for Coconut to work effectivelly. Hao et al. start with a CoT dataset, such as GSM8K, where questions, reasoning steps, and answers are clearly delimited. For a handful of epochs, the model is trained to predict the reasoning steps and answer, as in vanilla CoT. In each subsequent training stage, the first remaining reasoning step is replaced with a `<cot>` token (indicating "continuous thought") surrounded by `<bot>` and `<eot>` (for "beginning" and "end of thought") and the model is trained (via cross-entropy loss) to predict the subsequent reasoning steps and the answer. This is iterated until no more reasoning steps remain, and the model is merely tasked with predicting the answer.

> **Coconut Curriciulum**
>
> - [Question] [Step 1] [Step 2] ... [Step N] [Answer]
> - [Question] `<bot>` `<cot>` `<eot>` [Step 2] ... [Step N] [Answer]
> - [Question] `<bot>` `<cot>` ... `<cot>` `<eot>` [Step N] [Answer]
> - [Question] `<bot>` `<cot>` ... `<cot>` `<eot>` [Answer]

The decision to subsume reasoning steps left-to-right is a smart one: in doing so, the authors train the model to "sub-verbalize" steps in the order they occur. 

> **Aside**
>
> A few months after the Coconut paper, [Geiping et al. (2025)](https://arxiv.org/pdf/2502.05171) published work on using recurrent-depth LMs alongside latent reasoning to scale test-time compute. Their results are impressive, using a different architecture from Coconut and the other approaches considered in this post.



# Our Experimental Setup

The efficiency gains seen in the Coconut paper are impressive, needing often an order of magnitude fewer tokens than vanilla CoT to achieve greater performance, but going from, say, 92 to 9 additional tokens does not provide as significant a real-world speedup: the time to first token remains unchanged, and the unembed–re-embed step skipped by Coconut is not particularly intensive computationally relative to the rest of the transformer. Hence, we propose combining structured layer pruning with Coconut to leverage the efficiency gains of the former alongside the accuracy gains of the latter. 

Our hypothesis is that by first training a model on the Coconut curriculum, we teach it to reason continuously, freeing it from the restriction that it generate interpretable tokens diring each step. Thereafter, we can prune the model head-wise or layer-wise to lighten the computational load needed for continuous thought while retaining the capability of the model to utilize the continuous latent space afforded to it.

> **Aside**
> 
> Ultimately, our goal is to train a model to *flexibly* use more or less compute as it sees fit via some form of [routing or mixture-of-experts](https://arxiv.org/pdf/2409.14107). That way, models could decide to use only certain layers or attention heads in situations where less compute is needed, but choose to use the entire model when the extra compute would be particularly helpful. 

In our preliminary explorations of pruned Coconut, we work with Llama 3.2 1B, a state-of-the-art LM lightweight enough to run and be fine-tuned on a single GPU. We use the [Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) implementation of Llama 3.2 1B running on a single Nvidia A100 GPU.

## Overview of the Llama 3.2 1B architecture


Llama 3.2 1B is a decoder-only transformer, consisting of an embedding layer, a stack of 16 `LlamaDecoderLayer`s with layer normalization, and a final "LM head" projection layer. Each `LlamaDecoderLayer` contains a self-attention sublayer with 32 attention heads, a feed-forward sublayer `LlamaMLP` with `hidden_size=2048`, `intermediate_size=8192`, and `act_fn=silu`, giving a total of $2048 \times 8192 \times 2 + 8192 \times 2048 = 50,331,648$ parameters in the MLP sub-layer.


```python
class LlamaMLP(nn.Module):
   def __init__(self):
       super().__init__()
       self.gate_proj = nn.Linear(2048, 8192)
       self.up_proj = nn.Linear(2048, 8192)
       self.down_proj = nn.Linear(8192, 2048)


   def forward(self, x):
       down_proj = self.down_proj(silu(self.gate_proj(x)) * self.up_proj(x))
       return down_proj
```


Each of the 32 attention heads has query matrix of dimension $2048 \times 64$, key matrix of dimension $2048 \times 64$, and value matrix of dimension $2048 \times 64$, for a total of $2048 \times (64 \times 32) + 2048 \times (64 \times 8) + 2048 \times (64 \times 8) + 2048 \times (64 \times 32) = 12,582,912$ parameters in the attention sub-layer.


```python
LlamaAttention(
 (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
 (k_proj): Linear(in_features=2048, out_features=512, bias=False)
 (v_proj): Linear(in_features=2048, out_features=512, bias=False)
 (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
)
```
These combine to give a total of $16 \times (12,582,912 + 50,331,648) = 1,006,633,024$ parameters in the model.


> **Remark**
>
> The Llama 3.2 1B architecture utilizes a lightweight variant of multi-head attention called **Grouped Query Attention (GQA)** ([Ainslie et al. (2023)](https://arxiv.org/pdf/2305.13245)), wherein key and value matrices are shared across groups of query matrices. This can help greatly with the large memory requirements of multi-head attention, without enormous losses in performance.
> This explains why `q_proj` and `k_proj` have different dimensions in the `LlamaAttention` module: across 32 attention heads, there are 32 query matrices (with hidden dimension $2048 \div 32 = 64$), but only 8 key and value matrices (with hidden dimension $512 \div 8 = 64$).
>
> GQA can help greatly with the large memory requirements of multi-head attention, with only small losses in accuracy. Ainslie et al. (2023) show that GQA with $H / 8$ groups can reduce the time per sample by nearly 90% while losing marginal accuracy.

## Experiments to Run

While our work is still in its infancy, we plan to run a handful of preliminary experiments:
- training a model with the Coconut curriculum, removing layer(s) determined by some form of angular block distance or testing on a validation set, and healing the model
- doing the same with attention head pruning 
- combining the two in one direction or another
- use the various pruned models to explore **scaling laws for test-time compute**: what are the limits on accuracy of a reasoning model given a fixed amount of compute?

> **Goal**
>
> Train a Coconut model to flexibly choose to drop certain layers or attention heads at test-time, to balance accuracy with efficiency. 

It will be interesting to see how models could allocate either more layers or more tokens during reasoning. 