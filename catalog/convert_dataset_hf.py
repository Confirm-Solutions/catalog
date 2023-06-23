# Modified 2023 T. Ben Thompson, Confirm Solutions
# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import multiprocessing
import os
import warnings
from dataclasses import dataclass
from pprint import pprint
from typing import Dict, Iterable, Optional, Union

import datasets
import numpy as np
import typer
from datasets.download.streaming_download_manager import xexists, xglob, xopen
from streaming import MDSWriter
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from catalog.cli import dataclass_cli

# ------------------------------------------------------------------------------
# copied from llmfoundry:
# https://github.com/mosaicml/llm-foundry/blob/main/llmfoundry/data/datasets.py


class ConcatTokensDataset(IterableDataset):
    """An IterableDataset that returns token samples for MDSWriter.

    Returns dicts of {'tokens': bytes}

    To use data created by this class and written to MDS format:

    ```python
        import torch
        from streaming.base import StreamingDataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained('your/tokenizer')
        ds = StreamingDataset(local='mds-data-folder', split='val')

        # note, you need to copy the numpy array because the original is
        # non-writeable and torch does not support non-writeable tensors, so
        # you get a scary warning and if you do try to write to the tensor you
        # get undefined behavior
        tokens = torch.from_numpy(np.frombuffer(ds[0]['tokens'], dtype=np.int64).copy())
        print(tokenizer.decode(tokens))
    ```
    """

    def __init__(
        self,
        hf_dataset: Union[datasets.IterableDataset, datasets.Dataset],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        bos_text: str,
        eos_text: str,
        no_wrap: bool,
    ):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.max_length = max_length
        self.bos_text = bos_text
        self.eos_text = eos_text
        self.should_wrap = not no_wrap

        self.bos_tokens = self.tokenizer(
            self.bos_text, truncation=False, padding=False, add_special_tokens=False
        )["input_ids"]
        if len(self.bos_tokens) > 1:
            warnings.warn(
                "You specified --concat_tokens with --bos_text, but your"
                " BOS text is not tokenizing to one token "
                ", instead we got {self.bos_tokens}. Quit if this was in error."
            )

        self.eos_tokens = self.tokenizer(
            self.eos_text, truncation=False, padding=False, add_special_tokens=False
        )["input_ids"]
        if len(self.eos_tokens) > 1:
            warnings.warn(
                "You specified --concat_tokens with --eos_text, but your EOS text"
                " is not tokenizing to one token"
                ", instead we got {self.eos_tokens}. Quit if this was in error."
            )

        eos_text_provided = self.eos_text != ""
        bos_text_provided = self.bos_text != ""
        test_text = self.tokenizer("")
        if len(test_text["input_ids"]) > 0 and (eos_text_provided or bos_text_provided):
            message = (
                "both eos and bos"
                if eos_text_provided and bos_text_provided
                else ("eos_text" if eos_text_provided else "bos_text")
            )
            warnings.warn(
                "The provided tokenizer adds special tokens, but you also specified"
                f" {message}. This may result "
                "in duplicated special tokens. Please be sure this is what you intend."
            )

    def __iter__(self) -> Iterable[Dict[str, bytes]]:
        buffer = []
        for sample in self.hf_dataset:
            encoded = self.tokenizer(sample["text"], truncation=False, padding=False)
            iids = encoded["input_ids"]
            buffer = buffer + self.bos_tokens + iids + self.eos_tokens
            while len(buffer) >= self.max_length:
                concat_sample = buffer[: self.max_length]
                buffer = buffer[self.max_length :] if self.should_wrap else []
                yield {
                    # convert to bytes to store in MDS binary format
                    # NOTE: CONVERT TO UINT16, THIS COULD BREAK WITH OTHER
                    # TOKENIZERS
                    "tokens": np.asarray(concat_sample, dtype=np.uint16).tobytes()
                }


# ------------------------------------------------------------------------------
# based on
# https://github.com/mosaicml/llm-foundry/blob/main/scripts/data_prep/convert_dataset_hf.py


@dataclass
class Config:
    # job
    config: str = ""
    num_workers: int = None
    begin_shard: int = 0
    end_shard: int = -1
    n_samples_per_shard: int = None
    just_merge: bool = False
    skip_merge: bool = False

    # source data
    path: str = "EleutherAI/the_pile_deduplicated"
    name: str = None
    split: str = "train"
    streaming: bool = True

    # tokenization
    concat_tokens: int = 2048
    tokenizer: str = "EleutherAI/gpt-neox-20b"
    batch_size: int = 512

    # output
    out_root: str = "out"
    compression: str = "zstd:16"
    wrap: bool = True


def worker_wrapper(args):
    """Helper to unpack arguments for worker"""
    return worker(*args)


def worker(
    c: Config,
    job_idx: int,
    ds: datasets.Dataset = None,
):
    out_dir = os.path.join(c.out_root, str(job_idx))
    print(f"Checking {out_dir}", flush=True)
    if xexists(os.path.join(out_dir, "finished")):
        print(f"Skipping job {job_idx} because it is already finished", flush=True)
        return

    if ds is None:
        ds = datasets.load_dataset(c.path, c.name, split=c.split, streaming=c.streaming)
        # This is an ugly hack to focus on a particular shard. We just remove
        # the other shards from the iterator.
        iterable_kwargs = ds._ex_iterable.kwargs
        for k in ["archive_iterators", "filepaths", "files"]:
            if k in iterable_kwargs:
                iterable_kwargs[k] = iterable_kwargs[k][job_idx : job_idx + 1]

    tokenizer = AutoTokenizer.from_pretrained(c.tokenizer)
    # we will enforce length, so suppress warnings about sequences too long
    # for the model
    tokenizer.model_max_length = int(1e30)

    dataset = ConcatTokensDataset(
        hf_dataset=ds,
        tokenizer=tokenizer,
        max_length=c.concat_tokens,
        bos_text=tokenizer.bos_token,
        eos_text=tokenizer.eos_token,
        no_wrap=not c.wrap,
    )
    loader = DataLoader(dataset=dataset, sampler=None, batch_size=c.batch_size)
    sampler = generate_samples(loader)

    with MDSWriter(
        columns={"tokens": "bytes"},
        out=out_dir,
        compression=c.compression,
    ) as out:
        for i, sample in enumerate(sampler):
            if c.n_samples_per_shard is not None and i >= c.n_samples_per_shard:
                break
            out.write(sample)
    return job_idx


def generate_samples(
    loader: DataLoader, truncate_num_samples: Optional[int] = None
) -> Iterable[Dict[str, bytes]]:
    """Generator over samples of a dataloader.

    Args:
       loader (DataLoader): A dataloader emitting batches like
            {key: [sample0_bytes, sample1_bytes, sample2_bytes, ...]}
       truncate_num_samples (Optional[int]):
            An optional # of samples to stop at.

    Yields:
        Sample dicts.
    """
    n_samples = 0
    for batch in loader:
        keys = list(batch.keys())
        current_bs = len(batch[keys[0]])
        for idx in range(current_bs):
            if truncate_num_samples is not None and n_samples >= truncate_num_samples:
                return
            n_samples += 1
            yield {k: v[idx] for k, v in batch.items()}


# ------------------------------------------------------------------------------
# from https://github.com/mosaicml/streaming/blob/main/streaming/text/convert/pile.py#L145
# the original version moved the files, but that seems unnecessary. let's just
# leave the shards spread across subdirectories.


def merge_shard_groups(root: str) -> None:
    """Merge ephemeral sub-datasets created in parallel into one dataset.

    Args:
        root (str): Root directory.
    """
    import json

    subdirs = sorted(xglob(root + "/*"))
    infos = []
    missing = []
    for subdir in subdirs:
        if "index.json" in subdir:
            continue
        index_filename = os.path.join(subdir, "index.json")
        try:
            with xopen(index_filename) as f:
                obj = json.load(f)
        except FileNotFoundError:
            missing.append(subdir)
        just_subdir = os.path.basename(subdir)
        for info in obj["shards"]:
            for k in ["raw_data", "zip_data"]:
                info[k]["basename"] = os.path.join(just_subdir, info[k]["basename"])
            infos.append(info)
    if missing:
        raise ValueError(f"Missing {len(missing)} shards: {missing}")

    obj = {
        "version": 2,
        "shards": infos,
    }
    text = json.dumps(obj, sort_keys=True)
    out_filename = os.path.join(root, "index.json")
    print(f"writing to {out_filename} with {len(infos)} shards.")
    with xopen(out_filename, "w") as f:
        f.write(text)


# ------------------------------------------------------------------------------


@dataclass_cli
def main(c: Config) -> None:
    """Tokenize and convert a HuggingFace dataset to MDS format."""
    print("Config:")
    pprint(c)

    if c.num_workers is None:
        c.num_workers = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(c.num_workers)

    if not c.just_merge:
        dsb = datasets.load_dataset_builder(c.path, c.name)
        expected_num_samples = dsb.info.splits[c.split].num_examples
        print(f"{expected_num_samples=}")

        ds = datasets.load_dataset(c.path, c.name, split=c.split, streaming=c.streaming)

        if c.streaming:
            if c.end_shard == -1:
                c.end_shard = ds.n_shards
            c.end_shard = min(c.end_shard, ds.n_shards)

            args = [(c, i) for i in range(c.begin_shard, c.end_shard)]
            if c.num_workers == 1:
                results_iter = map(worker_wrapper, args)
            else:
                results_iter = pool.imap(worker_wrapper, args)

            for i in tqdm(
                results_iter,
                desc=c.path,
                total=c.end_shard - c.begin_shard,
                smoothing=0.0,
            ):
                if i is not None:
                    finished_file = os.path.join(c.out_root, str(i), "finished")
                    with xopen(finished_file, "w") as f:
                        f.write("done")
        else:
            worker(c, 0, ds=ds)

    # each worker above writes to a subdirectory, combine these into a single
    # dataset here
    if not c.skip_merge:
        merge_shard_groups(c.out_root)


if __name__ == "__main__":
    typer.run(main)
