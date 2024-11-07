import:py torch;
import:py torch.nn as nn;
import:py torch.optim as optim;
import:py from torch.nn {Transformer} 
import:py math;

#*
Defining individual nodes

1. MultiHeadAttention
2. PositionWiseFeedForward
3. PositionalEncoding
4. EncoderLayer

*#
""" This will be the output connected to all other operational nodes """
node EncoderLayer{
    has self_attn: pyobj;
    has feed_forward: pyobj;
    has x: pyobj;

    
}
""" This is connected to positional encoding"""
node MultiHeadAttention {
    has x: pyobj;
    has Q: pyobj = torch.zeros(10);
    has K: pyobj = torch.zeros(10);
    has V: pyobj = torch.zeros(10);
    
    has W_q: pyobj = nn.Linear(16, 16);  # Corresponds to W_q in PyTorch
    has W_k: pyobj = nn.Linear(16, 16);  # Corresponds to W_k
    has W_v: pyobj = nn.Linear(16, 16); 

    can apply_attention {
        # Perform scaled dot-product attention
        Q = W_q(x);  # Equivalent to PyTorch's W_q(Q)
        K = W_k(x);
        V = W_v(x);

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(16);  # Scaled dot-product
        attn_probs = torch.softmax(attn_scores, dim=-1);
        output = torch.matmul(attn_probs, V);
        
        x = W_o(output);  # Linear transformation as output
    }

}

node PositionWiseFeedForward {
    has x: pyobj;

    has fc1: pyobj = nn.Linear(16, 64);
    has fc2: pyobj = nn.Linear(64, 16);
    has activation: pyobj = nn.ReLU();

    can apply_ff {
        x = fc2(activation(fc1(x)));
    }
    
}


""" This is connected to input node"""
node PositionalEncoding {
    has x: pyobj;
    
}

node Input {
    has x: pyobj;

    
}



walker Creator {
    can traverse with `root entry {
        curr = here;
        batch_size = 2;       # Number of sequences in a batch
        seq_length = 5;       # Length of each sequence (number of tokens)
        d_model = 16;         # Dimensionality of each token embedding

        # Initialize input tensor
        input_tensor = torch.rand(batch_size, seq_length, d_model);
        input_node = Input(x=input_tensor);
        
        curr++>input_node;
        visit [-->];
        
    }
    
    can pos_encode with Input entry {
        # Perform positional encoding transformation
        print("In pos_encode",here);
        pos_node = PositionalEncoding(x=here.x);  # Create positional encoding node with input data
        here++>pos_node;
        visit [-->];  # Connect to PositionalEncoding node
    }

    can multi_head_attn with PositionalEncoding entry {
        # Apply multi-head attention transformation
        print("In mha",here);
        mha_node = MultiHeadAttention(x=here.x);  # Create multi-head attention node with positional encoded data
        here++>mha_node;  
        visit [-->];# Connect to MultiHeadAttention node
    }

    can p_w_FF with MultiHeadAttention entry {
        # Apply position-wise feed-forward network transformation
        print("In pwff",here);
        pwff_node = PositionWiseFeedForward(x=here.x);  # Create PositionWiseFeedForward node with attention output
        here++>pwff_node; 
        visit [-->]; # Connect to PositionWiseFeedForward node
    }

    can encode with PositionWiseFeedForward entry {
        # Apply encoder operations
        print("In encoder",here);
        encode_node = EncoderLayer(x=here.x);  # Create EncoderLayer node
        here++>encode_node;
        visit [-->];  # Connect to EncoderLayer node
    }

    can output with EncoderLayer entry {
        # Output the final transformed data
        print("Encoder Output:", here.x);  # Print the output data from the EncoderLayer node
    }
}




with entry {
    root spawn :> Creator;
}