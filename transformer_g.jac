import:py torch;
import:py torch.nn as nn;
import:py torch.optim as optim;
import:py torch.nn.functional as F;
import:py from torch.nn {Transformer}
import:py math;


"""
Node 1:
1. Linear transformation of input to produce Q, K, or V.
2. Reshaping and permuting the tensor for multi-head attention.
"""
node GENMAT:nn.Module: {

    has t_layer: pyobj;
    has t: pyobj;
    has batch_size: int;
    has seq_length: int;
    #has input_dim: int;
    has d_model: int;
    has num_heads: int;
    has head_dim: float;

    can init(x:pyobj,d_model: int,num_heads: int) {
        super.init();
        self.t_layer = nn.Linear(d_model, d_model);
        self.t = self.t_layer(x);
        (self.batch_size,self.seq_length,self.d_model) = x.size();
        self.d_model = d_model;
        self.num_heads = num_heads;
        self.head_dim = d_model // num_heads;
    }

    can transform with entry{
        self.t = self.t.reshape(self.batch_size,self.seq_length,self.num_heads, self.head_dim);
        print(f"q/k/v size: {self.t.size()}");
        self.t = self.t.permute(0,2,1,3);
        print(f"q/k/v size post permute: {self.t.size()}");
    }

}

"""
1. Compute scaled dot-product between Q and K.
2. Apply the optional attention mask.
3. Generate attention scores using softmax.
"""
node ATTEN:nn.Module: {
    has mask: pyobj;
    has attention: pyobj;

    can init(mask: pyobj){
        super.init();
        self.mask = mask;
        self.attention = None;
    }

    can transform with encoder_walker entry {
        print("Inside Node 2");
        d_k = here.q.size()[-1];
        scaled = torch.matmul(here.q,here.k.transpose(-2,-1)) / math.sqrt(d_k);# Transposing last 2 dim -> seq-length, head dim_size
        if self.mask is not None{
            scaled +=self.mask;
        }
        self.attention = F.softmax(scaled, dim=-1);
    }

}

"""
1. Perform matrix multiplication of attention scores and V.
2. Reshape and permute the resulting tensor back to its original shape.
3. Apply the final linear transformation.
"""
node SDP:nn.Module:{

    has attention: pyobj;
    has linear_layer: pyobj;
    has d_model: int;
    has out: pyobj;
    has num_heads: int;
    has head_dim: float;

    can init(attention: pyobj,v:pyobj,x:pyobj,d_model: int,num_heads: int){
        super.init();
        self.linear_layer = nn.Linear(d_model, d_model);
        self.attention = attention;
        (self.batch_size,self.seq_length,self.d_model) = x.size();
        self.d_model = d_model;
        self.num_heads = num_heads;
        self.head_dim = d_model // num_heads;
        self.out = None;

    }

    can transform with encoder_walker entry {

        values = torch.matmul(here.attention,here.v); # Take from V
        values = values.permute(0,2,1,3).reshape(self.batch_size, self.seq_length, self.num_heads * self.head_dim);
        print(f"values.size(): {values.size()}");
        self.out = self.linear_layer(values);
        print(f"out.size(): {self.out.size()}");
    }

}

node LNMN:nn.Module:{
    has parameters_shape: pyobj;
    has eps: float;
    has gamma: pyobj;
    has beta: pyobj;
    has out: pyobj;
    can init(parameters_shape: pyobj, eps: float = 1e-5){
        super.init();
        self.parameters_shape = parameters_shape;
        self.eps = eps;
        self.gamma = nn.Parameter(torch.ones(parameters_shape));
        self.beta =  nn.Parameter(torch.zeros(parameters_shape));
        self.out = None;
    }
    can transform with encoder_walker entry {
        #*
        Our inputs must be x + residual_x
        i.e x after performing multiheaded attention
        *#

        inputs = here.x;


        dims = [-(i+1) for i in range(len(self.parameters_shape))];
        mean = inputs.mean(dim=dims, keepdim = True);
        print(f"Mean ({mean.size()})");
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True);
        std = (var + self.eps).sqrt();
        print(f"Standard Deviation  ({std.size()})");
        y = (inputs - mean) / std;
        print(f"y: {y.size()}");
        self.out = self.gamma * y  + self.beta;
        print(f"self.gamma: {self.gamma.size()}, self.beta: {self.beta.size()}");
        print(f"out: {self.out.size()}");
    }
}

node PWFFN:nn.Module:{
    has linear1: pyobj;
    has linear2: pyobj;
    has relu: pyobj;
    has dropout: pyobj;
    has out: pyobj;
    can init(d_model: int,hidden: int, drop_prob: float = 0.1){
        super.init();
        self.linear1 = nn.Linear(d_model, hidden);
        self.linear2 = nn.Linear(hidden, d_model);
        self.relu = nn.ReLU();
        self.dropout = nn.Dropout(p=drop_prob);
        self.out = None;
    }

    can transform with encoder_walker entry {
        x = here.res_x;
        x = self.linear1(x);
        print(f"x after first linear layer: {x.size()}");
        x = self.relu(x);
        print(f"x after activation: {x.size()}");
        x = self.dropout(x);
        print(f"x after dropout: {x.size()}");
        x = self.linear2(x);
        print(f"x after 2nd linear layer: {x.size()}");
        self.out = x;
    }
}

node ADD:nn.Module:{
    has x: pyobj;
    can init()
    {
        super.init();
        self.x = None;
    }

    can store with encoder_walker entry {
        if(here.res_x!=None)
        {
            self.x = here.x + here.res_x;
            print("Added with residual value!!");
        }
    }
}

node RES:nn.Module:{
    has x: pyobj;
    can init()
    {
        super.init();
        self.x = None;
    }
    can store with encoder_walker entry {
        if(here.res_x!=None)
        {
            self.x = here.x;
        }
    }
}


can connect_encoder(x:pyobj,d_model: int,num_heads: int,ffn_hidden: int,drop_prob:float) -> node{
    (batch_size,seq_length,d_model) = x.size();
    mask = None;
    attention = None;
    v = None;
    # Nodes to be initialized
    query = GENMAT(x,d_model,num_heads); # Query
    key = GENMAT(x,d_model,num_heads); # Key
    value = GENMAT(x,d_model,num_heads); # Value
    node2 = ATTEN(mask); # Get attention
    node3 = SDP(attention,v,x,d_model,num_heads); # Scaled Dot Product and project back
    lnm_1 = LNMN([d_model]);
    lnm_2 = LNMN([d_model]);
    ffn = PWFFN(d_model,ffn_hidden, drop_prob);
    add_1 = ADD();
    add_2 = ADD();
    result = RES();


    # First we do Multiheaded addention
    qkv_list = [query,key,value];


    ## Correct order order of Encoder
    root++>qkv_list;
    qkv_list++>node2;
    node2++>node3; # Result of Multiheaded attention

    node3++>add_1; # Adding initial x with output of Multiheaded attention
    #root++>add_1;
    add_1++>lnm_1;
    lnm_1++>ffn;
    ffn++>add_2;
    add_2++>lnm_2;
    lnm_2++>result;
    return result;  


}


walker encoder_walker:nn.Module: {

    has x: pyobj; # stores input x at start
    has qkv_list: list;
    has q: pyobj;
    has k: pyobj;
    has v: pyobj;
    has attention: pyobj;
    # has mha_x: pyobj;
    has res_x: pyobj; # stores result x
    # has ffn_x: pyobj;

    can init(x:pyobj,d_model: int,num_heads: int) {
        super.init();
        self.qkv_list = [];
        self.q = None;
        self.k = None;
        self.v = None;
        self.x = x;
        self.res_x = None; #only set when we need result of multiheaded attention, Feed forward network
        print(f"x.size(): {x.size()}");
        #connect_mha(x,d_model,num_heads);
    }

    can traverse with `root entry {
        visit [-->];
    }

    can call_GENMAT with GENMAT exit {
        print("Visited Node in Node 1\n");
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

    can call_ATTEN with ATTEN exit {
        print("Visited Node in Node 2\n");
        self.attention = here.attention;
        #print("Attention after ATTEN",self.attention);
        visit[-->];

    }

    can call_SDP with SDP exit {
        print("Visited Node in Node 3\n");
        #print(here.out);
        self.res_x = here.out;
        visit[-->];
    }

    can call_layer_normalization with LNMN exit {
        print("Visited a Layer Normalization Node\n");
        #print(here.out);
        self.res_x = here.out;
        visit[-->];

    }

    can call_positionwise_feedforward with PWFFN exit {
        print("Visited Feed Forward Node\n");
        #print(here.out);
        self.res_x = here.out;
        visit[-->];

    }

    can call_add with ADD exit {
        print("Visited Add Node\n");
        #print(here.x);
        self.x = here.x;
        visit[-->];
    }


}


with entry {
    #input_dim = 1024;
    d_model = 512;
    num_heads = 8;
    batch_size = 30;
    seq_length = 200;
    ffn_hidden = 2048;
    drop_prob = 0.1;
    num_layers = 5;

    #*eq_length = 4;
    batch_size = 1;
    input_dim = 512;
    d_model = 512;
    num_heads = 8;*#

    #d_model = 2;
    #num_heads = 1;
    #batch_size = 4;
    #seq_length = 5;
    #ffn_hidden = 4;
    #drop_prob = 0.1;

    x = torch.rand((batch_size,seq_length,d_model));
    #for i=0 to i<num_layers by i+=1
    #{
    #    next = connect_encoder(x,d_model,num_heads,ffn_hidden,drop_prob);
    #}
    connect_encoder(x,d_model,num_heads,ffn_hidden,drop_prob);
    root spawn encoder_walker(x,d_model,num_heads);

}