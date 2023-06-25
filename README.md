# A catalog of several millions tasks Pythia can do.

Please read the main post at [https://confirmlabs.org/posts/catalog.html](https://confirmlabs.org/posts/catalog.html).

Further details and reproduction commands here.

The datasets:

- [pile_scan_4](https://huggingface.co/datasets/Confirm-Labs/pile_scan_4)
- [pile_bigrams](https://huggingface.co/datasets/Confirm-Labs/pile_bigrams)
- [pile_bigram_prefixes](https://huggingface.co/datasets/Confirm-Labs/pile_bigram_prefixes)
- [pile_top_bigrams](https://huggingface.co/datasets/Confirm-Labs/pile_top_bigrams)
- [pile_trigrams](https://huggingface.co/datasets/Confirm-Labs/pile_trigrams)
- [pile_trigram_prefixes](https://huggingface.co/datasets/Confirm-Labs/pile_trigram_prefixes)
- [pile_top_trigrams](https://huggingface.co/datasets/Confirm-Labs/pile_top_trigrams)


## Construction strategy

### Bigrams and trigrams

We use the deduplicated Pile. The data has been tokenized and pre-processed
into the MosaicML Streaming format (MDS). This was not necessary but I had
already done it for a separate project and found it convenient to be able to
stream the tokenized dataset rather than downloading the entire thing and
re-tokenizing it.

The `group by id0, id1, id2` operations necessary to construct bigrams/trigrams
are memory hungry. The default DuckDB `group by` operation ran out of memory so
I used a sharding strategy with Parquet files and DuckDB. This was probably
overengineering where I should've just used an instance with 3TB of memory, but
it works and it wasn't very hard to setup. Using DuckDB over other parquet
handling tools seemed to make sense because:
- It's very fast and uses memory more sparingly compared to Pandas.
- Many bulk parquet-processing tools like Spark are much more involved to
  setup. DuckDB is just a pip install.
- I eventually put the tables into a DuckDB format anyway for a nice SQL
  interface.

To perform the sharding, we stream The Pile from S3 and split into Parquet
files by `id0 % 64`. Separating into chunks by id0 makes the later
`group by id0, id1, id2` step simple to do out-of-core. 

### Prompt scanning

We run prompt scanning only on the first shard of
[`EleutherAI/the_pile_deduplicated`](https://huggingface.co/datasets/EleutherAI/the_pile_deduplicated).
There are 1650 shards total. Shard #1 contains ~112.5 million tokens. Running
on the first shard provides plenty of data for our current purposes but scaling
up to a larger portion of the dataset would be straightforward. I would
recommend using multiple GPUs because a single shard required ~19 hours to run
on a single A6000.

## Reproduction

Set up the environment:

```
mamba env create # or conda env create, but mamba is much faster
conda activate catalog
poetry install
```

Construct the [MDS format streaming
dataset](https://github.com/mosaicml/streaming) in S3. `out-root` can be set to
a local directory if you prefer.

```
python convert_dataset_hf.py --out-root="s3://caiplay/the_pile_deduplicated_16"
```

Run the bigram/trigram dataset construction code. On a `r6a.16xlarge` instance
with 64 cores and 512GB RAM, the total runtime is ~3 hours, dominated by step 1
at 2 hours. The total disk size required is ~1TB. The final tables are ~80 GB
in DuckDB format.

```
python catalog/build_ngrams.py all \
    --work-dir=data \
    --db-filename=data/ngrams.db \
    --batches 1000
```

Run prompt scanning on the first shard of The Pile. On a single A6000 GPU, the
total runtime is ~19 hours. 
```
python catalog/scanner.py all --work-dir=./data/scan_2.8b --batches 1000
```