import:py torch;
import:py torch.nn as nn;
import:py torch.optim as optim;
import:py torch.nn.functional as F;
import:py from torch.nn {Transformer}
import:py math;


"""
Describes a single encoder Pass
"""
node Encoder_layer {

    can init()
    {
    }

    can add_character with Encoder entry {
        here.encoder_string += "Hello";
        here.num_layers-=1;
    }

}

walker Encoder {

    has num_layers: int;
    has encoder_string: string;
    can init(num_layers: int){
        # Number of passes in incoder
        self.num_layers = num_layers;
        self.encoder_string = "";
    }

    can traverse with `root entry {
        visit[-->(`?Encoder_layer)];
    }

    can terminate with Encoder_layer exit{
        if (self.num_layers == 0)
        {
            print("Final String :",self.encoder_string);
        }
        else
        {
            visit[-->(`?Encoder_layer)];
        }
    }

}


with entry {
    num_layers = 5;
    #*
    Pre-Connecting each encoder layer node
    based on the number of layers
    *#
    encoder_node = Encoder_layer();
    root++>encoder_node;
    for i in range(num_layers)
    {
        new_encoder_node = Encoder_layer();
        encoder_node++>new_encoder_node;
        encoder_node = new_encoder_node;
    }
    root spawn Encoder(num_layers);
}
