

<br> nlp task to find word in sentence that corresponds to location, of span 1; so San Fancisco wont be recognised
<br> called word Window Classification
<br> data; usually .txt or .csv where each row corresponds to a sentence or tabular datapoint

<br> Our raw data, which consists of sentences
```
corpus = [
          "We always come to Paris",
          "The professor is from Australia",
          "I live in Stanford",
          "He comes from Taiwan",
          "The capital of Turkey is Ankara"
         ]
```
<br> preprocessing; steps: Tokenization, Lowercase, Noise Removal, Stop Word Removal; step depends on task
def preprocess_sentence(sentence):
  return sentence.lower().split()

<br> Create our training set
```
train_sentences = [sent.lower().split() for sent in corpus]
```

<br> also have labels; so our model wants to output 0 for words not in location and 1 for words that are
<br> Set of locations that appear in our corpus
```
locations = set(["australia", "ankara", "paris", "stanford", "taiwan", "turkey"])
```

<br> Our train labels
```
train_labels = [[1 if word in locations else 0 for word in sent] for sent in train_sentences]
```

<br> convert into embeddings; 
-  How to turn words into numbers; have embedding lookup table; word in ocabulary have embedding i in table
  1.  Find index `i` of word in embedding table: `word->index`
  2.  Index into embedding table `index->embedding`

Lookng at steps; can find unique words in corpus and assign an index to each. <br>
We can also add `<unk>` to tackle words outside of vocabulary; only requirement is that the string is unique; and only use
for unknown words; we add this to our vocabulary `vocabulary.add("<unk>")` <br>
Because our model may expect a fixed number of inputs; we can pad our sentences with `<pad>`; which we can add to our vocabulary.

<p>

Then we can convert training sentences into sequence of indices corresponding to each token <br>
Then with an index for each word; can create an embedding table `nn.Embedding`; `nn.Embedding(num_words, embedding_dimension)`; where num_words is number of words in vocabulary; and embedding dimensions is dimensions of embedding we want. This is just a wrapper around trainable `NxE` dimensional tensor; where N is number of words in vocab E; and E is number of embedding dimensions; random at first but changes over time. Updated through backpropagation. <p>
To get word embedding for a word in vocab; we create a lookup tensor; that is just a tensor with index we want to look up nn.Embedding; can do multiple embeddings at once <br>
Typically; we define the embedding layer as part of our model (see in later sections of notebook).
<br> 

For more stable updates; we can use a batch updater: `torch.util.data.DataLoader` class; `DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=collage_fn)`;
we can write a custom fn to collage_fn to print stats about batch or perform extraprocessing. <br> 
Can put all in one place; use `partial` python fnc to pass parameters we give to function we pass it; <br>
To make loops; can use `unfold(dimension, size, step)`; to make window dimensions