import:py torch;
import:py torch.nn as nn;
import:py torch.optim as optim;
import:py torch.nn.functional as F;
import:py from torch.nn {Transformer}
import:py math;

# Implementing mha in jaclang
can scaled_dot_product(q:pyobj,k:pyobj,v:pyobj,mask:pyobj=None) -> tuple {
    d_k = q.size()[-1];
    scaled = torch.matmul(q,k.transpose(-2,-1)) / math.sqrt(d_k);# Transposing last 2 dim -> seq-length, head dim_size
    if mask is not None{
        scaled +=mask;
    }
    attention = F.softmax(scaled, dim=1);
    values = torch.matmul(attention, v);
    return (values,attention);
}



node PositionalEncoding:nn.Module: {
    has max_seq_length: int;
    has d_model: int;
    can init(max_seq_length: int) {
        super.init();
        self.max_seq_length = max_seq_length;
        self.d_model = d_model;
    }

    can forward() -> pyobj {
        even_i = torch.arange(0, self.d_model, 2).float();
        denominator = torch.pow(10000, even_i/self.d_model);
        position = torch.arange(self.max_seq_length).reshape(self.max_seq_length, 1);
        even_PE = torch.sin(position / denominator);
        odd_PE = torch.cos(position / denominator);
        stacked = torch.stack([even_PE, odd_PE], dim=2);
        PE = torch.flatten(stacked, start_dim=1, end_dim=2);
        return PE;
    }
}

node MultiheadAttention:nn.Module: {

    has input_dim: int;
    has d_model: int;
    has num_heads: int;
    has head_dim: int;
    has qkv_layer: pyobj;
    has linear_layer: pyobj;

    can init() {
        super.init();
        self.input_dim = input_dim;
        self.d_model = d_model;
        self.num_heads = num_heads;
        self.head_dim = d_model // num_heads;
        self.qkv_layer = nn.Linear(input_dim, 3*d_model);
        self.linear_layer = nn.Linear(d_model,d_model);
    }

    can forward(x: pyobj,mask: pyobj=None) -> pyobj {
        (batch_size,seq_length,input_dim) = x.size();
        print(f"x.size: {x.size()}");
        qkv = self.qkv_layer(x);
        print(f"qkv.size: {qkv.size()}");
        qkv = qkv.reshape(batch_size,seq_length,self.num_heads,3*self.head_dim);
        print(f"qkv.size: {qkv.size()}");
        qkv = qkv.permute(0,2,1,3);
        print(f"qkv.size: {qkv.size()}");
        (q,k,v) = qkv.chunk(3,dim=-1); # 3 vectors - q,v,k
        print(f"q.size: {q.size()}");
        (values,attention) = scaled_dot_product(q,k,v,mask);
        print(f"values.size: {values.size()}");
        values = values.reshape(batch_size,seq_length,self.num_heads*self.head_dim);
        print(f"values.size: {values.size()}");
        out = self.linear_layer(values);
        print(f"out.size: {out.size()}");
        return out;
    }

}


walker PhaseWalker  {

    has max_seq_length: int;

    can init() {
        self.max_seq_length = 10;
    }
    can traverse with `root entry {

        pe_node = PositionalEncoding(self.max_seq_length);
        here ++> pe_node;
        visit [-->];
        #mha_node = MultiheadAttention();
        #here ++>mha_node;
        #visit [-->];
    }

    can call_mha with MultiheadAttention exit {
        mha_out = here.forward(x);
        print(mha_out);
    }

    can call_pe with PositionalEncoding exit {
        pe_out = here.forward();
        print(pe_out);
    }

}



with entry {
    seq_length = 5;
    batch_size = 30;
    input_dim = 1024;
    d_model = 512;
    num_heads = 8;

    x = torch.rand((batch_size,seq_length,input_dim));

    root spawn PhaseWalker();

}