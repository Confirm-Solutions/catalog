# A catalog of several millions tasks Pythia can do.

## Construction strategy

We use the deduplicated Pile. The data has been tokenized and pre-processed
into the MosaicML Streaming format (MDS). This was not necessary but I had
already done it for a separate project and found it convenient to be able to
stream the tokenized dataset rather than downloading the entire thing and
re-tokenizing it.

The `group by id0, id1, id2` operations necessary to construct bigrams/trigrams
are fairly memory hungry. The default DuckDB
`group by` operation ran out of memory so I used a sharding strategy with Parquet files and
DuckDB. This was probably overengineering where I should've just used an
instance with 3TB of memory, but it works and it wasn't very hard to setup.
Using DuckDB over other parquet handling tools seemed to make sense because:
- It's very fast and uses memory more sparingly compared to Pandas.
- Many bulk parquet-processing tools like Spark are much more involved to
  setup. DuckDB is just a pip install.
- I eventually put the tables into a DuckDB format anyway for a nice SQL
  interface.

To perform the sharding, we stream The Pile from S3 and split into Parquet
files by `id0 % 64`. Separating into chunks by id0 makes the later
`group by id0, id1, id2` step simple to do out-of-core. 

## Hardware/Performance

- On a r6a.16xlarge instance (64 cores, 512GB RAM), the total runtime is ~3
  hours, dominated by step 1 at 2 hours.
- The total disk size required is ~1TB.
- The final tables are ~80 GB in DuckDB format.

## Reproduction

Set up the environment:
```
mamba env create
poetry install
```

Construct the MDS format streaming dataset in S3. `out-root` can be set to a
local directory if you prefer.
```
python convert_dataset_hf.py --out-root="s3://.../the_pile_deduplicated_16"
```

Run the bigram/trigram dataset construction code. The `db-filename` path will
contain the DuckDB database.
```
python catalog/build_ngrams.py all \
    --work-dir=data \
    --db-filename=data/ngrams.db \
    --batches 1000
```

python catalog/input_scanner.py all --work-dir=./data/scan_2.8b --batches 1000

Posts:
- link to first dataset post.