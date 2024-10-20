import:py torch;
import:py torch.nn as nn;
import:py torch.optim as optim;
import:py from torch.nn {Transformer} 
import:py math;

#* Trying to Map encoder in Jac lang:

Logically have some nodes, connected by edges and traverse through walkers.


*#


node TokenNode {
    has word: str; 
    has embedding: list[float] = [];
    has pos_encoding: list[float] = [];
}


walker encoder {
   
    has pos_enc: list[float] = [];
    can init() -> None {
        self.W_q = nn.Linear(16, 16);
        self.W_k = nn.Linear(16, 16);
        self.W_v = nn.Linear(16, 16);
    }
    can embed_token(word: str) -> list[float] {
    return [ord(char) % 256 / 256.0 for char in word] + [0.0] * (16 - len(word));
    }

    
    can positional_encoding(seq_len: int, dim: int){
        self.pos_enc = [];
        for pos in range(seq_len) {
            for i in range(0, dim, 2) {
                self.pos_enc.append(math.sin(pos / (10000 ** (i / dim))));
                if i+1 < dim{
                    self.pos_enc.append(math.cos(pos / (10000 ** ((i + 1) / dim))));
                }
            }
        }
    }

    # Multi-Head Attention with PyTorch
    can multi_head_attention(embedding: list[float]) -> list[float] {
        embedding_tensor = torch.tensor(embedding).unsqueeze(0);
        Q = self.W_q(embedding_tensor);
        K = self.W_k(embedding_tensor);
        V = self.W_v(embedding_tensor);
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(len(embedding));
        attn_probs = torch.softmax(attn_scores, dim=-1);
        context = torch.matmul(attn_probs, V);
        return context.squeeze(0).tolist();
    }

    can encode with entry {
        here.embedding = self.embed_token(here.word);
        print("Embedding:", here.embedding);
        self.positional_encoding(len(here.word), 16);  # Example: 16-dim encoding
        here.pos_encoding = self.pos_enc;
        here.embedding = self.multi_head_attention(here.embedding);
        print("Word:", here.word);
        print("Embedding with attention:", here.embedding);
        print("Positional Encoding:", here.pos_encoding);
    }
}

with entry {
    node1 = TokenNode(word="hello");
    node2 = TokenNode(word="world");

    
    node1 spawn encoder();
    node2 spawn encoder();
}
