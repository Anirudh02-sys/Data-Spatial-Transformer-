import torch
import torch.nn as nn
import math
from transformer import EncoderLayer

# Define MultiHeadAttention, PositionWiseFeedForward, PositionalEncoding, and EncoderLayer here (copy the classes as provided).

# Define the input parameters for the Encoder Layer
batch_size = 2 # no of inputs 
seq_length = 5 # Number of tokens 
d_model = 16 # hidden_dim size 
num_heads = 4 # Number of attention heads 
d_ff = 32 # dimensionality of the feed forward layer
dropout = 0.1 # dropout 

# Create a random input tensor (batch_size, seq_length, d_model)
input_tensor = torch.rand(batch_size, seq_length, d_model)

# Instantiate the Encoder Layer
encoder_layer = EncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)

# Create a mask (if needed). Here we'll use None to ignore masking for simplicity.
mask = None

# Forward pass through the Encoder Layer
output = encoder_layer(input_tensor, mask)

# Print the input and output tensors
print("Input Tensor:")
print(input_tensor)
print("\nOutput Tensor:")
print(output)