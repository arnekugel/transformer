import numpy as np

CHARS = ["a", "b"]
def tokenize(s): return [CHARS.index(c) for c in s]
def untok(tok): return CHARS[tok]

def softmax(x):
  exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
  return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def linear(x, w, b):
  return x @ w + b

def attention(q, k, v, mask):
  return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v


def causal_self_attention(x, c_attn, c_proj):
  # qkv projections
  x = linear(x, **c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]

  # split into qkv
  q, k, v = np.split(x, 3, axis=-1)  # [n_seq, 3*n_embd] -> 3 of [n_seq, n_embd]

  # causal mask to hide future inputs from being attended to
  causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10  # [n_seq, n_seq]

  # perform causal self attention
  x = attention(q, k, v, causal_mask)  # [n_seq, n_embd] -> [n_seq, n_embd]

  # out projection
  x = linear(x, **c_proj)  # [n_seq, n_embd] @ [n_embd, n_embd] = [n_seq, n_embd]

  return x

# [n_seq, n_embd] -> [n_seq, n_embd]
def transformer_block(x, attn):
  x = x + causal_self_attention(x, **attn)
  return x

# [n_seq] -> [n_seq, n_vocab]
def gpt(inputs, wte, wpe, blocks):
  # token + positional embeddings
  x = wte[inputs] + wpe[range(len(inputs))]  # [n_seq] -> [n_seq, n_embd]

  # forward pass through n_layer transformer blocks
  for block in blocks:
    x = transformer_block(x, **block)  # [n_seq, n_embd] -> [n_seq, n_embd]

  # project to vocab
  return x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]