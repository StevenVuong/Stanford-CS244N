# NMT Assignment
Note: Heavily inspired by the https://github.com/pcyin/pytorch_nmt repository


Answer to 1g) <br>
step() sets the attention scores e_t to -inf where enc_masks has 1, i.e. where e_t is computed using 'pad' embedding.
So when we compute corresponding attention weights at using softmax, these scores come from 'pad' embeddings will become
exp(-inf)=0.
<br>
This is important as we only want real word entries; so filter away any contribution of 'pad' entries in a way, computing the 
attention weights at, since they will all become 0. The non-zero attention weights will only be computed for real tokens 
in the sequence.