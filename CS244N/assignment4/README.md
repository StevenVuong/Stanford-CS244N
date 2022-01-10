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
    <p>
-  2a) Cherokee is a polysynthetic language, ie. its words are composed of may morphmemes. The lengths of Cherokee words
    can vary a lot, and its vocabulary can be extremely large due to unlimited combinations and permutations or morphmemes (the smallest grammatical unit of speech). Therefore, modelling Cherokee-to-English NMT problem at morphmeme-level or subword-level makes sense; as opposed to word level
-  2b) The amount of characters/subwords are smaller than that of all the words, so there are less character/subword    embeddings than word embeddings
-  2c) Multilingual training can provide the model with more data than a single data-scarce language can proide. These are effective at generalization, and capable of capturing representational similarity across a large body of languages. Languages can transfer knowledge from each other.
-   2d) NMT System repeated the translation of the latter part of sentence in the middle of the translation. This error may have occured by placing an overly high attention to the latter part of the sentence. We can possibly fix this by adjusting the attention weights according to the location of the words. Or adding more recurrent layers for encoder to further encode inter-word relationship.
-  2dii) NMT picked wrong pronoun. Likely caused by final softmax output with vocabulary projection. A possible fix would be to add more layers to the final vocabulary projection.
-  2diii) NMT system has meaning correct but fail to express a special word. Error is likely caused by low memory power or representational capacity of the model. Possible fix is to add more layers to projection layers of encoder & decoder which are currently single-layer.
-  2fiii) Single reference translation may be biased by a single human translator's preference or habit. Not unbiased or fair representation of ood translation, so can be biased and not comprehensive.

