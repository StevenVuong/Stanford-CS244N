# Notes:
-  Co-occurance Matrix: how often things co-occur in some environment
    -  <START> <END> tokens to represent beginning & end of sentences; and include these for co-occurance counts.
-  Then next step is to run Singular Value Decomposition (SVD) to reduce dimensions (top k principal)
    -  This reduced-dimensionality representation preserves semantic relationship between words
    -  In reality; it is expensive to perform SVD for large corpora, so can perform Truncated SVD for a small k; then we can compute this reasonably efficiently at scale.
-  Cosine distance is a metric for how much space 'must be travelled' to get from one point to another; and can examine the angle between them as such

## Notes on Reading GloVE paper:
