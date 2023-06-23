import os
from dataclasses import dataclass

import datasets
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import transformers
import typer
from streaming import StreamingDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

import obvcomp.cli as cli


@dataclass
class Config:
    steps: str
    config: str = ""
    batches: int = 1
    batch_size: int = 2**7
    micro_batch_size: int = 2**9
    model_name: str = "EleutherAI/pythia-2.8b-deduped"
    tokenizer_name: str = "EleutherAI/gpt-neox-20b"
    work_dir: str = "data/scan_2.8b"
    seq_len: int = 4


class StreamingTextDataset(StreamingDataset):
    def __getitem__(self, idx: int):
        sample = super().__getitem__(idx)
        return torch.from_numpy(
            np.frombuffer(sample["tokens"], dtype=np.uint16).astype(np.int32)
        )


def get_model(name: str):
    return (
        transformers.GPTNeoXForCausalLM.from_pretrained(
            name, low_cpu_mem_usage=True, torch_dtype=torch.float16
        )
        .cuda()
        .eval()
    )


def step1_scan(cfg: Config):
    print("Loading dataset")
    dataset = StreamingTextDataset(
        remote="s3://caiplay/the_pile_deduplicated_16/0",
        local="./pile_tmp",
        cache_limit="50gb",
        keep_zip=False,
        batch_size=cfg.batch_size,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=2,
        prefetch_factor=2,
        pin_memory=True,
        persistent_workers=True,
    )
    n_batch = int(np.ceil(len(dataset) / cfg.batch_size))

    model = get_model(cfg.model_name)
    os.makedirs(cfg.work_dir, exist_ok=True)

    n_iters = min(n_batch, cfg.batches)
    for i, samples in tqdm(enumerate(dataloader), total=n_iters):
        out_fp = os.path.join(cfg.work_dir, f"scan_{cfg.seq_len}_batch_{i}.pt")
        if os.path.exists(out_fp):
            continue

        if cfg.batches is not None and i >= cfg.batches:
            break
        samples = samples.cuda()
        short = samples[:, 1:].unfold(1, cfg.seq_len, 1)
        long = samples.unfold(1, cfg.seq_len + 1, 1)
        short_2d = short.reshape(-1, cfg.seq_len).cuda()
        long_2d = long.reshape(-1, cfg.seq_len + 1).cuda()

        js = torch.empty(short_2d.shape[0], dtype=torch.float32, device="cuda")
        logit_excite_max = torch.empty(
            short_2d.shape[0], dtype=torch.float32, device="cuda"
        )
        logit_inhibit_max = torch.empty(
            short_2d.shape[0], dtype=torch.float32, device="cuda"
        )
        pmax_delta = torch.empty(short_2d.shape[0], dtype=torch.float32, device="cuda")
        pmax_short = torch.empty(short_2d.shape[0], dtype=torch.float32, device="cuda")
        pmax_long = torch.empty(short_2d.shape[0], dtype=torch.float32, device="cuda")

        with torch.no_grad():
            for j in tqdm(range(0, short_2d.shape[0], cfg.micro_batch_size)):
                j_end = min(j + cfg.micro_batch_size, short_2d.shape[0])
                short_logits = model(short_2d[j:j_end]).logits[:, -1]
                long_logits = model(long_2d[j:j_end]).logits[:, -1]

                short_p = torch.softmax(short_logits, dim=-1)
                long_p = torch.softmax(long_logits, dim=-1)
                short_ls = torch.log_softmax(short_logits, dim=-1)
                long_ls = torch.log_softmax(long_logits, dim=-1)
                kl_s = (short_p * (short_ls - long_ls)).sum(axis=1)
                kl_l = (long_p * (long_ls - short_ls)).sum(axis=1)
                js[j:j_end] = (kl_s + kl_l) / 2

                logit_excite_max[j:j_end] = (
                    (long_logits - short_logits).max(dim=-1).values
                )
                logit_inhibit_max[j:j_end] = (
                    (short_logits - long_logits).max(dim=-1).values
                )
                pmax_delta[j:j_end] = (short_p - long_p).abs().max(dim=-1).values
                pmax_short[j:j_end] = short_p.max(dim=-1).values
                pmax_long[j:j_end] = long_p.max(dim=-1).values

        torch.save(
            [
                samples,
                js,
                logit_excite_max,
                logit_inhibit_max,
                pmax_delta,
                pmax_short,
                pmax_long,
            ],
            out_fp,
        )


def step2_scan_filter(cfg: Config):
    model = get_model(cfg.model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.tokenizer_name)

    batch_filenames = os.listdir(cfg.work_dir)
    batch_ids = [int(os.path.splitext(f)[0].split("_")[-1]) for f in batch_filenames]
    batch_ids.sort()
    n_batches = max(batch_ids) + 1

    examples = []
    for batch_idx in tqdm(range(n_batches)):
        [
            samples,
            js,
            logit_excite_max,
            logit_inhibit_max,
            pmax_delta,
            pmax_short,
            pmax_long,
        ] = torch.load(
            os.path.join(cfg.work_dir, f"scan_{cfg.seq_len}_batch_{batch_idx}.pt"),
            map_location="cpu",
        )

        js = js.numpy()
        logit_excite_max = logit_excite_max.numpy()
        logit_inhibit_max = logit_inhibit_max.numpy()
        pmax_delta = pmax_delta.numpy()
        pmax_short = pmax_short.numpy()
        pmax_long = pmax_long.numpy()

        samples = samples.cuda()
        short = samples[:, 1:].unfold(1, cfg.seq_len, 1)
        long = samples.unfold(1, cfg.seq_len + 1, 1)
        short_2d = short.reshape(-1, cfg.seq_len).cuda()
        long_2d = long.reshape(-1, cfg.seq_len + 1).cuda()

        def get_context(i, k=0):
            div = i // short.shape[1]
            mod = i % short.shape[1]
            return samples[div, mod - k + 1 : mod + cfg.seq_len + 1 + k]

        assert (get_context(100000, 0) == short_2d[100000]).all()

        high_post = np.where((pmax_long > 0.5) & (js > 0.5))[0]
        with torch.no_grad():
            for k_start in range(0, high_post.shape[0], cfg.micro_batch_size):
                i_batch = high_post[k_start : k_start + cfg.micro_batch_size]
                short_batch = short_2d[i_batch]
                long_batch = long_2d[i_batch]
                short_logits = model(short_batch).logits[:, -1]
                long_logits = model(long_batch).logits[:, -1]
                short_p = torch.softmax(short_logits, dim=-1)
                long_p = torch.softmax(long_logits, dim=-1)

                long_batch_np = long_batch.cpu().numpy()
                short_p.max(dim=-1).values.to(torch.float32).cpu().numpy()
                short_max_id = short_p.argmax(dim=-1).cpu().numpy()
                long_p.max(dim=-1).values.to(torch.float32).cpu().numpy()
                long_max_id = long_p.argmax(dim=-1).cpu().numpy()
                context = tokenizer.batch_decode([get_context(i, k=5) for i in i_batch])
                examples.append(
                    dict(
                        i=i_batch,
                        token_short=tokenizer.batch_decode(short_max_id),
                        token_long=tokenizer.batch_decode(long_max_id),
                        p_short=pmax_short[i_batch],
                        p_long=pmax_long[i_batch],
                        JS=js[i_batch],
                        long_ids=[
                            long_batch_np[row] for row in range(long_batch_np.shape[0])
                        ],
                        short_max_id=short_max_id,
                        long_max_id=long_max_id,
                        context=context,
                        p_delta_max=pmax_delta[i_batch],
                        logit_excite_max=logit_excite_max[i_batch],
                        logit_inhibit_max=logit_inhibit_max[i_batch],
                    )
                )
    df = pd.concat([pd.DataFrame(e) for e in examples])
    df["text"] = df["long_ids"].apply(
        lambda x: "[" + tokenizer.decode(x[0]) + "]" + tokenizer.decode(x[1:])
    )
    df = (
        df.drop_duplicates(subset=["text"])
        .sort_values(by="p_long", ascending=False)
        .reset_index(drop=True)
    )

    df.to_parquet(os.path.join(cfg.work_dir, os.pardir, "scan_filter.parquet"))


def step3_huggingface(cfg: Config):
    dset = datasets.Dataset(
        pq.read_table(
            os.path.join(cfg.work_dir, os.pardir, "scan_filter.parquet"),
            memory_map=True,
        )
    )
    dset.push_to_hub(f"Confirm-Labs/pile_scan_{cfg.seq_len}", split="scan")


@cli.dataclass_cli
def main(cfg: Config):
    """
    Scan The Pile for prefixes that result in large changes in model
    predictions.
    """
    print(cfg)
    cli.run_steps(step_dict, cfg.steps, cfg)


step_dict = cli.collect_steps(globals(), main)


if __name__ == "__main__":
    typer.run(main)
