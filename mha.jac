import:py torch;
import:py torch.nn as nn;
import:py torch.optim as optim;
import:py torch.nn.functional as F;
import:py from torch.nn {Transformer}
import:py math;


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

"""
Node 1: Take 1 copy ( Q/K/V )
"""
node Node1:nn.Module: {

    has t_layer: pyobj;
    has t: pyobj;
    has batch_size: int;
    has seq_length: int;
    has input_dim: int;
    has d_model: int;
    has num_heads: int;
    has head_dim: float;

    can init(x:pyobj,d_model: int,num_heads: int) {
        super.init();
        self.t_layer = nn.Linear(input_dim, d_model);
        self.t = self.t_layer(x);
        (self.batch_size,self.seq_length,self.input_dim) = x.size();
        self.d_model = d_model;
        self.num_heads = num_heads;
        self.head_dim = d_model // num_heads;
    }

    can transform with entry{
        self.t = self.t.reshape(self.batch_size,self.seq_length,self.num_heads, self.head_dim);
        self.t = self.t.permute(0,2,1,3);
    }

}


node Node2:nn.Module: {
    has mask: pyobj;
    has attention: pyobj;

    can init(mask: pyobj){
        super.init();
        self.mask = mask;
        self.attention = None;
    }

    can transform with MHA_Walker entry {
        print("Inside Node 2");
        d_k = here.q.size()[-1];
        scaled = torch.matmul(here.q,here.k.transpose(-2,-1)) / math.sqrt(d_k);# Transposing last 2 dim -> seq-length, head dim_size
        if self.mask is not None{
            scaled +=self.mask;
        }
        self.attention = F.softmax(scaled, dim=1);
    }

}

node Node3:nn.Module:{
    has attention: pyobj;
    has v: pyobj;
    has linear_layer: pyobj;
    has d_model: int;
    has out: pyobj;

    can init(attention: pyobj,v:pyobj,x:pyobj,d_model: int,num_heads: int){
        super.init();
        self.linear_layer = nn.Linear(d_model, d_model);
        self.attention = attention;
        self.v = v;
        (self.batch_size,self.seq_length,self.input_dim) = x.size();
        self.d_model = d_model;
        self.num_heads = num_heads;
        self.head_dim = d_model // num_heads;
        self.out = None;

    }

    can transform with MHA_Walker entry {
        values = torch.matmul(here.attention,here.v); # Take from V
        values = values.permute(0,2,1,3).reshape(self.batch_size, self.seq_length, self.num_heads * self.head_dim);
        self.out = self.linear_layer(values);
    }

}

walker MHA_Walker:nn.Module: {

    has x:pyobj;
    has batch_size: int;
    has seq_length: int;
    has input_dim: int;
    has d_model: int;
    has num_heads: int;
    has qkv_list: list;
    has q: pyobj;
    has k: pyobj;
    has v: pyobj;
    has attention: pyobj;

    can init(x:pyobj,d_model: int,num_heads: int) {
        super.init();
        self.x = x;
        (self.batch_size,self.seq_length,self.input_dim) = x.size();
        self.d_model = d_model;
        self.num_heads = num_heads;
        self.qkv_list = [];
        self.q = None;
        self.k = None;
        self.v = None;
    }

    can traverse with `root entry {
        visit [-->];
    }

    can call_node1 with Node1 exit {
        print("Visiting Node in Node 1: ", here);
        self.qkv_list.append(here.t);
        print(len(self.qkv_list));
        if(len(self.qkv_list) == 3)
        {
            self.q = self.qkv_list[0];
            self.k = self.qkv_list[1];
            self.v = self.qkv_list[2];
            visit[-->];
        }

    }

    can call_node2 with Node2 exit {
        print("Visiting Node in Node 2:", here);
        self.attention = here.attention;
        print("Attention after Node2",self.attention);
        visit[-->];

    }

    can call_node3 with Node3 exit {
        print("Visiting Node in Node 3:");
        print(here.out);
    }


}


can connect_mha(x:pyobj,d_model: int,num_heads: int){


    (batch_size,seq_length,input_dim) = x.size();
    mask = None;
    attention = None;
    v = None;
    # Nodes to be initialized
    node1_1 = Node1(x,d_model,num_heads); # Query
    node1_2 = Node1(x,d_model,num_heads); # Key
    node1_3 = Node1(x,d_model,num_heads); # Value
    node2 = Node2(mask); # Get attention
    node3 = Node3(attention,v,x,d_model,num_heads); # Scaled Dot Product and project back

    # Connections made
    root++>node1_1;
    root++>node1_2;
    root++>node1_3;
    node1_1++>node2;
    node1_2++>node2;
    node1_3++>node2;
    node2++>node3;

}




with entry {
    #*input_dim = 1024;
    d_model = 512;
    num_heads = 8;
    batch_size = 30;
    seq_length = 5;*#

    seq_length = 4;
    batch_size = 1;
    input_dim = 512;
    d_model = 512;
    num_heads = 8;

    x = torch.rand((batch_size,seq_length,input_dim));

    connect_mha(x,d_model,num_heads);
    root spawn MHA_Walker(x,d_model,num_heads);

}