- `id0`: the first token in the trigram
- `id1`: the second token in the trigram
- `id2`: the most common token following `(id0, id1)` in The Pile
- `sum_count`: the number of times that `(id0, id1)` appears in The Pile.
- `max_count`: the number of times that `id2` appears after `(id0, id1)` in The Pile.
- `frac_max`: `max_count / sum_count`
- `token0`: the string representation of `id0`
- `token1`: the string representation of `id1`
- `token2`: the string representation of `id2`
- `seq`: the string representation of the trigram, `token0 token1 token2`
- `p_{model_size}`: the probability of the trigram under Pythia-{model_size} when prompted with `(id0, id1)`.