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
    
    can embed_token(word: str) -> list[float] {
        return [ord(char) % 256 / 256.0 for char in word];
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

    can encode with entry {
        here.embedding = self.embed_token(here.word);
        self.positional_encoding(len(here.word), 16);  # Example: 16-dim encoding
        here.pos_encoding = self.pos_enc;
        print("Word:", here.word);
        print("Embedding:", here.embedding);
        print("Positional Encoding:", here.pos_encoding);
    }
}

with entry {
    node1 = TokenNode(word="hello");
    node2 = TokenNode(word="world");

    node1 spawn encoder();
    node2 spawn encoder();
}
