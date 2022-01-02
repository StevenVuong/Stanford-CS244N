# Notes:
-  Co-occurance Matrix: how often things co-occur in some environment
    -  <START> <END> tokens to represent beginning & end of sentences; and include these for co-occurance counts.
-  Then next step is to run Singular Value Decomposition (SVD) to reduce dimensions (top k principal)
    -  This reduced-dimensionality representation preserves semantic relationship between words
    -  In reality; it is expensive to perform SVD for large corpora, so can perform Truncated SVD for a small k; then we can compute this reasonably efficiently at scale.
-  Cosine distance is a metric for how much space 'must be travelled' to get from one point to another; and can examine the angle between them as such

## Notes on Reading GloVE paper:
Link: https://nlp.stanford.edu/pubs/glove.pdf

-  Combines matrix factorization & local context window with log-bilinear regression model
-  Evaluation scheme favours models that produce dimensions of meaning; `queen=man-woman`
-  Co-occurace probabilities; so find that probabilities between word corpora is large for where relevant
    -  Model vector differences to try to achieve co-occurance probabilities
    -  Adding bias can restore symmetry
-  Xi = sum of X; number of times any word appears in the context of word i. And Pij = P(j|i) = Xij/X; probability word j appears in context of word i