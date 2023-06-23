from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import transformers

_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = transformers.AutoTokenizer.from_pretrained(
            "EleutherAI/gpt-neox-20b"
        )
    return _tokenizer


def get_token_id(token):
    return get_tokenizer().encode(token)


########################################
# Model helpers
########################################


@dataclass
class ModelInfo:
    name: str
    param_count: float

    def get_hf_model(self):
        with torch.device("cuda"):
            model = (
                transformers.GPTNeoXForCausalLM.from_pretrained(
                    self.name, low_cpu_mem_usage=True, torch_dtype=torch.float16
                )
                .cuda()
                .eval()
            )
        return model


valid_param_str = ["70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"]


def model_cfg(param_str):
    assert param_str in valid_param_str
    return ModelInfo(
        name=f"EleutherAI/pythia-{param_str}-deduped",
        param_count=float(param_str[:-1]) * (1e6 if param_str[-1] == "m" else 1e9),
    )


def predict_last_token(model, S, k=10):
    S_encoded = get_tokenizer().encode(S)
    S_minus_token = get_tokenizer().decode(S_encoded[:-1])
    p, top_df = predict_simple(model, S_minus_token, key=k)

    return dict(
        prompt=S_minus_token,
        correct_id=S_encoded[-1],
        correct_token=get_tokenizer().decode(S_encoded[-1]),
        correct_p=p[S_encoded[-1]],
        top_k_df=top_df,
    )


def predict_simple(model, S, k=10):
    model.eval()
    inputs = get_tokenizer()(S, return_tensors="pt")
    if model.device.type == "cuda":
        for key in inputs:
            inputs[key] = inputs[key].cuda()
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
    p = torch.nn.functional.softmax(logits, dim=0).detach().cpu().numpy()

    top_df = pd.DataFrame(dict(top_k=p.argsort()[::-1][:k]))
    top_df["p"] = p[top_df["top_k"]]
    top_df["token"] = [_tokenizer.decode([i]) for i in top_df["top_k"]]
    return p, top_df


########################################
# Database lookup helpers
########################################


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


def lookup_trigram_prefix(id0, id1, db):
    df = db.query(
        f"""
        select trigrams.*,
               CAST(CAST(count as decimal)
                    / trigram_prefixes.sum_count as float) as frac
            from trigrams
            join trigram_prefixes
                on trigrams.id0 = trigram_prefixes.id0
                and trigrams.id1 = trigram_prefixes.id1
            where trigrams.id0 = {id0} and trigrams.id1 = {id1}
            order by trigrams.count desc
    """
    ).df()
    df = add_tokens(df, db)
    return df


def lookup_bigram_prefix(id0, db):
    db.query("select * from tokens").df()
    df = db.query(
        f"""
        select bigrams.*,
               CAST(CAST(count as decimal)
                    / bigram_prefixes.sum_count as float) as frac
            from bigrams
            join bigram_prefixes
                on bigrams.id0 = bigram_prefixes.id0
            where bigrams.id0 = {id0}
            order by bigrams.count desc
    """
    ).df()
    df = add_tokens(df, db)
    return df


def lookup_bigram_suffix(id1, db):
    db.query("select * from tokens").df()
    df = db.query(
        f"""
        with bigram_suffixes as (
            select sum(count) as sum_count
                from bigrams
                where id1 = {id1}
        )
        select bigrams.*,
               CAST(CAST(count as decimal)
                    / bigram_suffixes.sum_count as float) as frac
            from bigrams, bigram_suffixes
            where bigrams.id1 = {id1}
            order by bigrams.count desc
    """
    ).df()
    df = add_tokens(df, db)
    return df


########################################
# Interpretation functions
########################################


def plot_resid_norm(model, prompts, prepend_bos):
    logits, cache = model.run_with_cache(prompts, prepend_bos=prepend_bos)
    logits = logits[:, -1]
    resid = cache.accumulated_resid(layer=-1, pos_slice=-1)
    torch.softmax(logits, dim=-1)
    scaling = torch.norm(resid, dim=-1)
    plt.plot(scaling.cpu().numpy())
    plt.xlabel("layer")
    plt.ylabel("residual stream norm")
    plt.show()


@dataclass
class Lenses:
    prompts: List[str]
    tokenizer: transformers.PreTrainedTokenizer

    logit_lens_logits: torch.Tensor
    logit_lens_p: torch.Tensor
    logit_lens_top_ids: torch.Tensor
    logit_lens_top_tokens: torch.Tensor

    tuned_lens_logits: torch.Tensor
    tuned_lens_p: torch.Tensor
    tuned_lens_top_ids: torch.Tensor
    tuned_lens_top_tokens: torch.Tensor

    def view(self, idx, true_id):
        logit_rank = (
            self.logit_lens_p.shape[-1]
            - torch.where(self.logit_lens_p[:, idx].argsort(dim=1) == true_id)[1]
        )
        tuned_rank = (
            self.tuned_lens_p.shape[-1]
            - torch.where(self.tuned_lens_p[:, idx].argsort(dim=1) == true_id)[1]
        )
        n_layers = self.logit_lens_logits.shape[0]
        return pd.DataFrame(
            dict(
                layer=np.arange(n_layers),
                true=np.full_like(
                    self.logit_lens_top_tokens[:, idx],
                    self.tokenizer.decode(true_id),
                ),
                tuned=self.tuned_lens_top_tokens[:, idx],
                logit=self.logit_lens_top_tokens[:, idx],
                tuned_p=self.tuned_lens_p[:, idx, true_id].cpu().numpy(),
                logit_p=self.logit_lens_p[:, idx, true_id].cpu().numpy(),
                tuned_rank=tuned_rank.cpu().numpy(),
                logit_rank=logit_rank.cpu().numpy(),
            )
        )

    def plot_true_trajectory(self, true_id, true_p):
        true_id = true_id.astype(np.int32)
        n_prompts = self.logit_lens_logits.shape[1]
        print(n_prompts)
        for idx in range(n_prompts):
            true_token = self.tokenizer.decode(true_id[idx]).replace(" ", "_")
            p_data = true_p[idx]
            prompt_str = "".join(
                [
                    "[" + self.tokenizer.decode(id_).replace(" ", "_") + "]"
                    for id_ in self.tokenizer.encode(self.prompts[idx])
                ]
            )
            plt.subplot(n_prompts, 2, 1 + 2 * idx)
            plt.title(f"{prompt_str}[{true_token}] (p = {p_data:.3f})")
            plt.plot(
                self.tuned_lens_logits[:, idx, true_id[idx]].cpu().numpy(),
                "--",
                label="tuned",
            )
            plt.plot(
                self.logit_lens_logits[:, idx, true_id[idx]].cpu().numpy(),
                "-",
                label="logit",
            )
            plt.legend()
            plt.ylim([-5, 30])
            plt.xlabel("layer")
            plt.ylabel("logit[target]")
            plt.subplot(n_prompts, 2, 2 + 2 * idx)
            plt.plot(
                self.tuned_lens_p[:, idx, true_id[idx]].cpu().numpy(),
                "--",
                label="tuned",
            )
            plt.plot(
                self.logit_lens_p[:, idx, true_id[idx]].cpu().numpy(),
                "-",
                label="logit",
            )
            plt.legend()
            plt.ylim([0, 1])
            plt.xlabel("layer")
            plt.ylabel("p[target]")
        plt.show()


def run_lenses(model, hf_model, prompts, prepend_bos):
    logits, cache = model.run_with_cache(prompts, prepend_bos=prepend_bos)
    logits = logits[:, -1]
    resid = cache.accumulated_resid(layer=-1, pos_slice=-1)

    # NOTE: apply_ln_to_stack uses the cached layer norm scalings from the
    # run_with_cache call above. Freezing the layer norm scaling in this way makes
    # sense when trying to decompose the outputs into a linear combination of the
    # various MLPs/attn heads. But, it's not clear that this is the right thing to
    # do with logit lens since we're just cutting the remainder of the model off.
    # In this case, it probably makes more sense to recompute layer norm scalings.
    # Recomputing scalings *does* produce better looking outputs.
    #
    # This version re-uses the cached scalings:
    logit_lens_logits = model.unembed(
        cache.apply_ln_to_stack(resid, layer=-1, pos_slice=-1)
    )
    # this version recomputes scalings:
    # logit_lens_logits = torch.stack(
    #     [
    #         model.unembed(model.ln_final(resid[i].unsqueeze(1)))[:, -1]
    #         for i in range(resid.shape[0])
    #     ]
    # )
    logit_lens_p = torch.softmax(logit_lens_logits, dim=-1)
    torch.testing.assert_close(logit_lens_logits[-1], logits, rtol=1e-3, atol=1e-3)

    ll_top_ids = logit_lens_logits.argmax(dim=-1)
    ll_top_tokens = np.array(model.to_str_tokens(ll_top_ids.flatten())).reshape(
        ll_top_ids.shape
    )

    from tuned_lens.nn.lenses import TunedLens

    e_tuned_lens = TunedLens.from_model_and_pretrained(hf_model).to("cuda")
    tuned_lens_logits = torch.stack(
        [e_tuned_lens(resid[i], i) for i in range(model.cfg.n_layers)]
    )

    # append the final logits so that we can plot trajectory all the way through
    tuned_lens_logits = torch.cat((tuned_lens_logits, logits.unsqueeze(0)), dim=0)
    tuned_lens_p = torch.softmax(tuned_lens_logits, dim=-1)
    tuned_top_ids = tuned_lens_logits.argmax(dim=-1)
    tuned_top_tokens = np.array(model.to_str_tokens(tuned_top_ids.flatten())).reshape(
        tuned_top_ids.shape
    )

    return Lenses(
        prompts=prompts,
        tokenizer=model.tokenizer,
        logit_lens_logits=logit_lens_logits,
        logit_lens_p=logit_lens_p,
        logit_lens_top_ids=ll_top_ids,
        logit_lens_top_tokens=ll_top_tokens,
        tuned_lens_logits=tuned_lens_logits,
        tuned_lens_p=tuned_lens_p,
        tuned_lens_top_ids=tuned_top_ids,
        tuned_lens_top_tokens=tuned_top_tokens,
    )
