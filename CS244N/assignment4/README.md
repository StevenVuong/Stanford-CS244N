# NMT Assignment
Note: Help from https://github.com/pcyin/pytorch_nmt repository

Notes:
-  torch.nn.embeddings is a ookup table that stores embedding of a fixed dictionary and size
    -  Specify size of dictionary of embeddings, and size of each embedding; and padding index; 
    which do not contribute to the gradient, so is not updated during training
    -  Can call on source sentence (padded) to embed; then rnn pack padded sentence to 'pack' pad that
-  torch.nn.LSTM; multi-layer long short term memory RNN to an input sequence
    -  input size; hidden size; default stacks = 2, and can configure bidirectional
-  torch.nn.LSTMCell; individual LSTM; wth bias
-  torch.nn.linear -> Linear transformation, y=xA^T + b; in_features, out_features; can do a concatenation and apply linear layer.
-  torch.split: splits tensor into chunks; which dimension and size of chunks
-  torch.cat: concatenate torch tensors; these must have the same shape (except in concatenating dimensions) or be empty
-  torch.bmm: batch matrix multiplication of matrices 
-  torch unsqueeze; returns a new tensor with dimension size 1 inserted at speciic position

Answers to Written Questions:
-  1g) <br>
    step() sets the attention scores e_t to -inf where enc_masks has 1, i.e. where e_t is computed using 'pad' embedding.
    So when we compute corresponding attention weights at using softmax, these scores come from 'pad' embeddings will become
    exp(-inf)=0.
    <br>
    This is important as we only want real word entries; so filter away any contribution of 'pad' entries in a way, computing the 
    attention weights at, since they will all become 0. The non-zero attention weights will only be computed for real tokens 
    in the sequence.
-  1hi) <br>
    Dot product attention is faster to compute and requires less parameters <br>
    However it is less expressive, the representational power is lower due to lack of parameters.
-  1hii) <br>
    Additive attention can freely control the dimension of the attention scores; making it more versatile for certain application. <br>
    One disadvantage is that it can require more parameters and be less efficient to compute than multiplicative attention.