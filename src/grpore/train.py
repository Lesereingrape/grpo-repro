"""Training loops for three policy-optimization arms on one env.

- ``grpo``      : group-relative advantages, PPO-style clipped surrogate,
                  KL(β) pull toward a frozen reference policy.
- ``reinforce`` : same network and rollouts, but a running-mean baseline and
                  no clipping / no KL term — the plain policy-gradient floor.
- ``dpo``       : offline. We freeze a reference, generate a preference set
                  from its own rollouts (better reward = chosen), and train
                  the direct preference objective. No online rollouts.

Every arm logs the same metric row so the ablation table compares
apples to apples: mean reward, mean episode length, KL to reference,
clip fraction, and policy entropy.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .adv import group_relative, mean_baseline
from .env import Prompt, reward, sample_prompts
from .policy import Policy, prompt_vector, tokens_to_str

#: Iteration budget the committed ``results/ablation.json`` curves were measured
#: at, and therefore what a bare ``grpore train`` should reproduce.
PUBLISHED_ITERS = 120


@dataclass
class RunConfig:
    algo: str = "grpo"
    seed: int = 0
    iters: int = PUBLISHED_ITERS
    prompts_per_iter: int = 16
    group_size: int = 8          # GRPO rollouts per prompt
    inner_epochs: int = 2        # PPO-style reuse of each rollout batch
    lr: float = 3e-3
    clip: float = 0.2
    beta: float = 0.02           # KL coefficient toward reference
    hidden: int = 48
    eval_every: int = 10
    eval_prompts: int = 128
    extra: dict = field(default_factory=dict)


@dataclass
class Rollout:
    prompt: Prompt
    ids: list[int]
    old_logp: float              # sequence logprob at collection time
    adv: float
    tokens: list[str] = field(default_factory=list)


def _make_pools(cfg: RunConfig) -> tuple[list[Prompt], list[Prompt]]:
    train_pool = sample_prompts(cfg.prompts_per_iter * cfg.iters, seed=cfg.seed + 1)
    eval_pool = sample_prompts(cfg.eval_prompts, seed=9001)
    return train_pool, eval_pool


def _rollout_group(policy: Policy, prompts: list[Prompt], group: int) -> list[Rollout]:
    """Collect ``group`` completions per prompt with rewards and old logprobs."""
    out: list[Rollout] = []
    for p in prompts:
        pv = prompt_vector(p)
        for _ in range(group):
            ids, lps = policy.generate(pv)
            tokens = tokens_to_str(ids).split()
            out.append(
                Rollout(
                    prompt=p,
                    ids=ids,
                    old_logp=sum(lps),
                    adv=0.0,  # filled by the caller once the group is known
                    tokens=tokens,
                )
            )
    return out


def _attach_grpo_advantages(rollouts: list[Rollout], group: int) -> None:
    # rollouts are ordered prompt-by-prompt, group per prompt
    for i in range(0, len(rollouts), group):
        chunk = rollouts[i : i + group]
        advs = group_relative([_rw(rl) for rl in chunk])
        for rl, a in zip(chunk, advs, strict=True):
            rl.adv = a


def _rw(rl: Rollout) -> float:
    return reward(rl.tokens, rl.prompt)


def _seq_logprob(policy: Policy, rl: Rollout) -> torch.Tensor:
    pv = prompt_vector(rl.prompt)
    return policy.token_logprobs(pv, rl.ids).sum()


def _policy_loss(policy, ref, rollouts, cfg, opt) -> dict:
    """Clipped surrogate + KL for GRPO over one inner epoch."""
    total = torch.zeros(())
    clip_hits = 0
    kls = []
    for rl in rollouts:
        lp = _seq_logprob(policy, rl)
        with torch.no_grad():
            old = torch.tensor(rl.old_logp)
            ref_lp = _seq_logprob(ref, rl)
        ratio = torch.exp(lp - old)
        clipped = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip)
        obj = torch.minimum(ratio * rl.adv, clipped * rl.adv)
        kl = (lp - ref_lp).item()
        kls.append(kl)
        total = total + (-obj + cfg.beta * kl)
        if (ratio - 1).abs().item() > cfg.clip:
            clip_hits += 1
    opt.zero_grad()
    total.backward()
    opt.step()
    n = max(1, len(rollouts))
    return {
        "clip_frac": clip_hits / n,
        "kl_ref": sum(kls) / n,
    }


def evaluate(policy: Policy, pool: list[Prompt]) -> dict:
    policy.eval()
    correct = 0
    rw = 0.0
    lengths = []
    for p in pool:
        pv = prompt_vector(p)
        ids, _ = policy.generate(pv)
        tokens = tokens_to_str(ids).split()
        r = reward(tokens, p)
        rw += r
        lengths.append(len(ids))
        if r == 1.0:
            correct += 1
    policy.train()
    n = len(pool)
    return {
        "reward": rw / n,
        "acc": correct / n,
        "mean_len": sum(lengths) / n,
    }


def train_grpo(cfg: RunConfig) -> list[dict]:
    torch.manual_seed(cfg.seed)
    policy = Policy(hidden=cfg.hidden)
    ref = copy.deepcopy(policy).eval()
    for prm in ref.parameters():
        prm.requires_grad_(False)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    train_pool, eval_pool = _make_pools(cfg)
    log: list[dict] = []
    for it in range(cfg.iters):
        start = (it * cfg.prompts_per_iter) % max(1, len(train_pool) - cfg.prompts_per_iter)
        batch = train_pool[start : start + cfg.prompts_per_iter]
        rollouts = _rollout_group(policy, batch, cfg.group_size)
        _attach_grpo_advantages(rollouts, cfg.group_size)
        metrics = {
            "iter": it,
            "algo": "grpo",
            "group_reward": _mean_rw(rollouts),
            "mean_len": _mean_len(rollouts),
        }
        for _ in range(cfg.inner_epochs):
            metrics.update(_policy_loss(policy, ref, rollouts, cfg, opt))
        metrics["entropy"] = _entropy(policy, batch)
        if (it + 1) % cfg.eval_every == 0 or it == 0:
            metrics["eval"] = evaluate(policy, eval_pool)
        log.append(metrics)
    return log


def train_reinforce(cfg: RunConfig) -> list[dict]:
    torch.manual_seed(cfg.seed)
    policy = Policy(hidden=cfg.hidden)
    ref = copy.deepcopy(policy).eval()
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    train_pool, eval_pool = _make_pools(cfg)
    log: list[dict] = []
    baseline = 0.0
    for it in range(cfg.iters):
        start = (it * cfg.prompts_per_iter) % max(1, len(train_pool) - cfg.prompts_per_iter)
        batch = train_pool[start : start + cfg.prompts_per_iter]
        rollouts = _rollout_group(policy, batch, group=1)
        rws = [_rw(rl) for rl in rollouts]
        advs, baseline = mean_baseline(rws, baseline)
        for rl, a in zip(rollouts, advs, strict=True):
            rl.adv = a
        loss = torch.zeros(())
        for rl in rollouts:
            lp = _seq_logprob(policy, rl)
            loss = loss - lp * rl.adv
        opt.zero_grad()
        loss.backward()
        opt.step()
        metrics = {
            "iter": it,
            "algo": "reinforce",
            "group_reward": _mean_rw(rollouts),
            "mean_len": _mean_len(rollouts),
            "clip_frac": 0.0,
            "kl_ref": sum((_seq_logprob(policy, r).item() - _seq_logprob(ref, r).item())
                          for r in rollouts) / max(1, len(rollouts)),
            "entropy": _entropy(policy, batch),
        }
        if (it + 1) % cfg.eval_every == 0 or it == 0:
            metrics["eval"] = evaluate(policy, eval_pool)
        log.append(metrics)
    return log


def _mean_rw(rollouts: list[Rollout]) -> float:
    return sum(_rw(r) for r in rollouts) / max(1, len(rollouts))


def _mean_len(rollouts: list[Rollout]) -> float:
    return sum(len(r.ids) for r in rollouts) / max(1, len(rollouts))


def _build_preferences(policy: Policy, prompts: list[Prompt], group: int) -> list[tuple]:
    """Offline pairs from a frozen model's own rollouts: chosen = best reward,
    rejected = worst, kept only when they differ so the pair carries signal."""
    pairs = []
    for p in prompts:
        pv = prompt_vector(p)
        cands = []
        for _ in range(group):
            ids, _ = policy.generate(pv)
            tokens = tokens_to_str(ids).split()
            cands.append((reward(tokens, p), ids))
        cands.sort(key=lambda x: x[0])
        lo, hi = cands[0], cands[-1]
        if hi[0] > lo[0]:
            pairs.append((p, hi[1], lo[1]))
    return pairs


def train_dpo(cfg: RunConfig) -> list[dict]:
    torch.manual_seed(cfg.seed)
    policy = Policy(hidden=cfg.hidden)
    ref = copy.deepcopy(policy).eval()
    for prm in ref.parameters():
        prm.requires_grad_(False)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    train_pool, eval_pool = _make_pools(cfg)
    # One offline preference set from the initial (reference) policy.
    pairs = _build_preferences(ref, train_pool[: cfg.prompts_per_iter * 4], cfg.group_size)
    log: list[dict] = []
    beta = 0.1
    for it in range(cfg.iters):
        if pairs:
            loss = torch.zeros(())
            for p, chosen, rejected in pairs:
                pol_c = _seq_lp_ids(policy, p, chosen)
                pol_r = _seq_lp_ids(policy, p, rejected)
                with torch.no_grad():
                    ref_c = _seq_lp_ids(ref, p, chosen)
                    ref_r = _seq_lp_ids(ref, p, rejected)
                logits = beta * ((pol_c - ref_c) - (pol_r - ref_r))
                loss = loss - F.logsigmoid(logits)
            loss = loss / len(pairs)
            opt.zero_grad()
            loss.backward()
            opt.step()
        metrics = {"iter": it, "algo": "dpo", "group_reward": _rw_mean_pairs(policy, pairs),
                   "mean_len": 0.0, "clip_frac": 0.0, "kl_ref": 0.0,
                   "n_pairs": len(pairs)}
        if (it + 1) % cfg.eval_every == 0 or it == 0:
            metrics["eval"] = evaluate(policy, eval_pool)
        log.append(metrics)
    return log


def _seq_lp_ids(policy: Policy, p: Prompt, ids: list[int]) -> torch.Tensor:
    return policy.token_logprobs(prompt_vector(p), ids).sum()


def _rw_mean_pairs(policy: Policy, pairs: list[tuple]) -> float:
    if not pairs:
        return 0.0
    tot = 0.0
    for p, chosen, _ in pairs:
        tokens = tokens_to_str(chosen).split()
        tot += reward(tokens, p)
    return tot / len(pairs)


def _entropy(policy: Policy, prompts: list[Prompt]) -> float:
    """Mean first-token entropy — a cheap, stable drift probe."""
    ent = 0.0
    for p in prompts:
        pv = prompt_vector(p).reshape(1, -1)
        h = torch.zeros(1, policy.hidden)
        x = torch.tensor([[0]], dtype=torch.long)
        inp = torch.cat([policy.embed(x).squeeze(0), pv], dim=-1)
        h = policy.gru(inp, h)
        probs = F.softmax(policy.head(h), dim=-1)
        ent += -(probs * (probs + 1e-12).log()).sum().item()
    return ent / max(1, len(prompts))


def run_algo(cfg: RunConfig) -> list[dict]:
    if cfg.algo == "grpo":
        return train_grpo(cfg)
    if cfg.algo == "reinforce":
        return train_reinforce(cfg)
    if cfg.algo == "dpo":
        return train_dpo(cfg)
    raise ValueError(f"unknown algo: {cfg.algo}")


def to_jsonl(log: list[dict]) -> str:
    return "\n".join(json.dumps(row, sort_keys=True) for row in log)
