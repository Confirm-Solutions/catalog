"""
"""
import os
import time
from dataclasses import dataclass

import datasets
import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import transformers
import typer
from torch.utils.data import DataLoader
from tqdm import tqdm

import catalog.cli as cli


@dataclass
class Config:
    steps: str
    config: str = ""
    id0_chunks: int = 64
    work_dir: str = "test"
    db_filename: str = "test/test.db"
    batches: int = 1
    vocab_size: int = 50432


def step1_download(cfg: Config):
    from streaming import StreamingDataset

    """
    Stream The Pile from S3 and split into Parquet files by `id0 % id0_chunks`
    (typically `id0 % 64`). Separating into chunks by id0 makes the later
    `group by id0, id1, id2` step simple to do out-of-core. The default DuckDB
    group by operation was running out of memory for me.

    The resulting Parquet files are in "{cfg.work_dir}/explode".
    """
    batch_size = 2**17
    seq_len = 2048

    print("Loading dataset")
    dataset = StreamingDataset(
        remote="s3://caiplay/the_pile_deduplicated_16",
        local="./pile_tmp",
        cache_limit="50gb",
        keep_zip=False,
        batch_size=batch_size,
        predownload=batch_size * 2,
    )

    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=12, prefetch_factor=2
    )
    n_batch = int(np.ceil(len(dataset) / batch_size))

    db = duckdb.connect()
    db.execute("SET enable_progress_bar=false;")

    print(f"Extracting n-grams for {cfg.batches}/{n_batch} batches")
    for i, samples in tqdm(enumerate(dataloader), total=min(n_batch, cfg.batches)):
        if cfg.batches is not None and i >= cfg.batches:
            break

        n_samples = len(samples["tokens"])

        ids = np.empty((n_samples, seq_len), dtype=np.uint16)
        for j in range(n_samples):
            ids[j, :] = np.frombuffer(samples["tokens"][j], dtype=np.uint16)

        for ng in [2, 3]:
            os.makedirs(os.path.join(cfg.work_dir, "explode", str(ng)), exist_ok=True)

            # for example: for trigrams, we want to get:
            # id_arrs=[ids[:, :-2], ids[:, 1:-1], ids[:, 2:]]
            id_arrs = []
            for k in range(ng):
                end_idx = -(ng - 1 - k) if k < ng - 1 else None
                id_arrs.append(ids[:, k:end_idx])
            ngrams_arr = np.stack(id_arrs, axis=2).reshape((-1, ng))
            pd.DataFrame(ngrams_arr, columns=[f"id{k}" for k in range(ng)])
            id_str = ",".join([f"id{k}" for k in range(ng)])
            db.execute(
                f"""
                COPY (
                    select
                        {id_str},
                        count(*) as count,
                        id0 % {cfg.id0_chunks} as id0_chunk
                    from ngram_df
                    group by {id_str}
                ) TO '{cfg.work_dir}/explode/{ng}' (
                """
                + """
                    FORMAT PARQUET,
                    PARTITION_BY (id0_chunk),
                    OVERWRITE_OR_IGNORE,
                    FILENAME_PATTERN "data_{uuid}"
                );
                """
            )


def step2_bigrams(cfg: Config):
    """
    Combine the Parquet bigram chunks --> `bigrams_raw`.

    If step 2 fails because there are too many open files, try running `ulimit
    -n 100000` and then relaunching.

    Took 10 minutes.
    """
    db = duckdb.connect(cfg.db_filename)
    db.execute(
        """
        create or replace table bigrams_raw
            (id0 usmallint, id1 usmallint, count uinteger)
        """
    )
    for i in tqdm(range(cfg.id0_chunks)):
        db.execute(
            f"""
            insert into bigrams_raw
                select id0, id1, CAST(sum(count) as UINTEGER) as count
                from read_parquet('{cfg.work_dir}/explode/2/id0_chunk={i}/*.parquet')
                group by id0, id1
            """
        )


def step3_clean_bigrams(cfg: Config):
    """
    - group by id0, id1 to get rid of duplicate bigram stats --> `bigrams`.
    - compute the most common suffix for each prefix --> `bigram_prefixes`.

    Took 40 seconds
    """
    db = duckdb.connect(cfg.db_filename)
    db.execute(
        """
        create or replace table bigrams as
            select id0, id1, CAST(sum(count) as UINTEGER) as count
            from bigrams_raw
            group by id0, id1
        """
    )
    db.execute(
        """
        CREATE OR REPLACE TABLE bigram_prefixes AS
        SELECT
            id0,
            ARG_MAX(id1, count) as id1,
            CAST(SUM(count) AS UBIGINT) AS sum_count,
            MAX(count) AS max_count,
            CAST(MAX(count) AS DECIMAL) / CAST(SUM(count) AS DECIMAL) AS frac_max
        FROM bigrams
        GROUP BY id0
        ORDER BY frac_max DESC
        """
    )


def recombine_shards(cfg, i):
    parquet_glob = f"{cfg.work_dir}/explode/3/id0_chunk={i}/*.parquet"
    out_fp = f"{cfg.work_dir}/recombined/id0_chunk={i}/data.parquet"
    os.makedirs(os.path.dirname(out_fp), exist_ok=True)
    db = duckdb.connect()
    db.execute(
        f"""
        COPY (
            select id0, id1, id2, CAST(sum(count) as UINTEGER) as count
                from read_parquet('{parquet_glob}')
                group by id0, id1, id2
        ) TO '{out_fp}' (
            FORMAT PARQUET
        );
        """
    )


def step4_recombine_shards(cfg: Config):
    """
    Combine the exploded Parquet trigram chunks into a single Parquet file per
    id0_chunk value. After this step, there should be `cfg.id0_chunks` Parquet
    files in the `{cfg.work_dir}/recombined` directory

    Took ~25 minutes.
    """
    for i in tqdm(range(cfg.id0_chunks)):
        recombine_shards(cfg, i)


def step5_trigrams(cfg: Config):
    """
    - Merge the `{cfg.work_dir}/recombined` Parquet files --> `trigrams_{i}`
    - Compute the most common suffix for each id0, id1 prefix -->
      `trigram_prefixes`

    Took ~10 minutes.
    """
    db = duckdb.connect(cfg.db_filename)
    for i in tqdm(range(cfg.id0_chunks)):
        db.execute(
            f"""
            create or replace table trigrams_{i} as
                SELECT id0, id1, id2, count
                FROM
                read_parquet('{cfg.work_dir}/recombined/id0_chunk={i}/data.parquet')
            """
        )
    db.execute(
        """
        CREATE OR REPLACE TABLE trigram_prefixes (
            id0 usmallint,
            id1 usmallint,
            id2 usmallint,
            sum_count ubigint,
            max_count uinteger,
            frac_max double
        )
        """
    )
    for i in tqdm(range(cfg.id0_chunks)):
        db.execute(
            f"""
            INSERT INTO trigram_prefixes
            SELECT
                id0,
                id1,
                ARG_MAX(id2, count) as id2,
                CAST(SUM(count) AS UBIGINT) AS sum_count,
                MAX(count) AS max_count,
                CAST(MAX(count) AS DECIMAL)
                    / CAST(SUM(count) AS DECIMAL) AS frac_max
            FROM read_parquet('{cfg.work_dir}/recombined/id0_chunk={i}/data.parquet')
            GROUP BY (id0, id1)
            ORDER BY frac_max DESC
            """
        )
    db.execute(
        """
        CREATE OR REPLACE TABLE trigrams (
            id0 usmallint,
            id1 usmallint,
            id2 usmallint,
            count uinteger
        )
        """
    )
    for i in tqdm(range(cfg.id0_chunks)):
        db.execute(f"""INSERT INTO trigrams SELECT * FROM trigrams_{i}""")

def add_tokens(df, db):
    if not isinstance(df, pd.DataFrame):
        df = df.df()
    tokens_df = db.query("select * from tokens").df()
    d = max([int(c[2:]) for c in df.columns if c.startswith("id")]) + 1
    for i in range(d):
        df[f"token{i}"] = tokens_df.loc[df[f"id{i}"], "token"].values
    df["seq"] = df["token0"]
    for i in range(1, d):
        df["seq"] += df[f"token{i}"]
    return df


def step6_dataset(cfg: Config):
    db = duckdb.connect(cfg.db_filename)

    tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    pd.DataFrame(
        dict(
            id=np.arange(cfg.vocab_size),
            token=[tokenizer.decode(i) for i in range(cfg.vocab_size)],
        )
    )
    db.execute("create or replace table tokens as select * from tokens_df")

    # Construct bigram dataset consisting of the prefixes that are most
    # predictive of suffix:
    # - the predictiveness must be > X%
    threshold = 0.5
    top_bi_df = db.query(
        f"select * from bigram_prefixes where frac_max > {threshold}"
    ).df()
    top_bi_df = add_tokens(top_bi_df, db)
    top_bi_df.to_parquet(os.path.join(cfg.work_dir, "top_bigrams.parquet"))

    # Construct trigram dataset consisting of the (id0, id1) prefixes that are
    # most predictive of id2:
    # - the predictiveness must be > X%
    # - also, the bigram prefix must appear at least 100 times.
    top_tri_df = db.query(
        f"""
        select * from trigram_prefixes
            where sum_count > 1000 and frac_max > {threshold}
            order by frac_max desc
    """
    ).df()
    top_tri_df = add_tokens(top_tri_df, db)
    top_tri_df.to_parquet(os.path.join(cfg.work_dir, "top_trigrams.parquet"))

def predict_all_models(df):
    for batch_size, param_str in [
        (1024, "70m"),
        (1024, "160m"),
        (512, "410m"),
        (512, "1b"),
        (512, "1.4b"),
        (512, "2.8b"),
        (256, "6.9b"),
        (128, "12b"),
    ]:
        print("starting", model_info.name)
        start = time.time()
        with torch.device("cuda"):
            model = (
                transformers.GPTNeoXForCausalLM.from_pretrained(
                    f"EleutherAI/pythia-{param_str}-deduped", low_cpu_mem_usage=True, torch_dtype=torch.float16
                )
                .cuda()
                .eval()
            )
        print(f"Loading model took {time.time() - start}")

        start = time.time()

        if "id2" not in df.columns:
            in_cols = ["id0"]
        else:
            in_cols = ["id0", "id1"]
        input_ids = torch.asarray(df[in_cols].values.astype(np.int32))
        # prepend BOS token?
        # Pythia does not add BOS tokens at all:
        # https://discord.com/channels/729741769192767510/938462108721483787/1116111104251265234
        if model.device.type == "cuda":
            input_ids = input_ids.cuda()

        logits = None
        correct_col = "id2" if "id2" in df.columns else "id1"
        p = np.empty(df.shape[0])
        with torch.no_grad():
            for i in tqdm(range(0, input_ids.shape[0], batch_size)):
                logits = model(input_ids[i : i + batch_size]).logits[:, -1].detach()
                correct_id = df.iloc[i : i + batch_size][correct_col].values.astype(
                    np.int32
                )
                p[i : i + batch_size] = (
                    torch.nn.functional.softmax(logits, dim=1)[
                        torch.arange(logits.shape[0]), correct_id
                    ]
                    .cpu()
                    .to(torch.float32)
                    .numpy()
                )
        df[f"p_{param_str}"] = p
        print("Running model took", time.time() - start)


def step7_inference(cfg: Config):
    """
    Add the probabilities of the correct token to the top_bigrams and
    top_trigrams tables.
    """
    df = pd.read_parquet(os.path.join(cfg.work_dir, "top_bigrams.parquet"))
    df = df.sort_values(by="sum_count", ascending=False).reset_index(drop=True)
    predict_all_models(df)
    df.to_parquet(os.path.join(cfg.work_dir, "top_bigrams_p.parquet"))

    df = pd.read_parquet(os.path.join(cfg.work_dir, "top_trigrams.parquet"))
    df = df.sort_values(by="sum_count", ascending=False).reset_index(drop=True)
    predict_all_models(df)
    df.to_parquet(os.path.join(cfg.work_dir, "top_trigrams_p.parquet"))


def duckdb_to_hf(cfg, table):
    db = duckdb.connect(cfg.db_filename, read_only=True)
    db.execute(
        f"COPY {table} to '{cfg.work_dir}/to_hf/{table}.parquet' (FORMAT PARQUET)"
    )
    dset = datasets.Dataset(
        pq.read_table(f"{cfg.work_dir}/to_hf/{table}.parquet", memory_map=True)
    )
    dset.push_to_hub(f"Confirm-Labs/pile_{table}", split=table)


def step8_huggingface(cfg: Config):
    """
    Push tables to huggingface
    """
    pass
    # os.makedirs(f"{cfg.work_dir}/to_hf", exist_ok=True)

    # for fn in ["top_bigrams", "top_trigrams"]:
    #     dset = datasets.Dataset(
    #         pq.read_table(f"{cfg.work_dir}/{fn}_p.parquet", memory_map=True)
    #     )
    #     dset.push_to_hub(f"Confirm-Labs/pile_{fn}", split=fn)

    # tables = ["bigrams", "bigram_prefixes", "trigram_prefixes"]
    # pool = mp.Pool()
    # pool.starmap(duckdb_to_hf, [(cfg, t) for t in tables])

    # # trigrams is really big, so we first write a partitioned parquet
    # db = duckdb.connect(cfg.db_filename, read_only=True)
    # for i in tqdm(range(64)):
    #     db.execute(
    #         f"""
    #         COPY
    #             (select *, CAST(id0 % 64 as INT) as id0_chunk from trigrams_{i})
    #         TO '{cfg.work_dir}/to_hf/trigrams'
    #         """
    #         """
    #         (
    #             FORMAT PARQUET,
    #             PARTITION_BY id0_chunk,
    #             OVERWRITE_OR_IGNORE,
    #             FILENAME_PATTERN "file_{uuid}"
    #         )
    #         """
    #     )
    # # fortunately, datasets lets us use partitioned parquet files:
    # # This upload takes a long time! ~1 hour
    # dset = datasets.Dataset.from_parquet(
    #     f"{cfg.work_dir}/to_hf/trigrams/**/*", num_proc=10
    # )
    # dset.push_to_hub("Confirm-Labs/pile_trigrams", split="trigrams")


@cli.dataclass_cli
def main(cfg: Config):
    """
    Construct bigram/trigram completion tables from The Pile.
    """
    print(cfg)
    cli.run_steps(step_dict, cfg.steps, cfg)


step_dict = cli.collect_steps(globals(), main)

if __name__ == "__main__":
    typer.run(main)
