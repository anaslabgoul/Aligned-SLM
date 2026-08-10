"""
Train a model with Direct Preference Optimization (DPO) on verifiable-math pairs.

DPO (Rafailov et al., 2023) aligns a policy to preferences without a reward model
or RL. Starting from the SFT checkpoint, it raises the log-probability of the
"chosen" chain relative to the "rejected" one, while a *frozen copy* of the same
checkpoint (the reference) keeps the policy from drifting. The loss is

    L = -E[ log sigmoid( beta * ( (logp_pi(chosen)  - logp_ref(chosen))
                                 -(logp_pi(rejected) - logp_ref(rejected)) ) ) ]

where each logp is the sum of per-character log-probs over the *response* tokens
only (the prompt is masked out). beta controls how hard the reference leash pulls.

Run generate_dpo_data.py first to produce the preference pairs.

Examples
--------
Align model_2 on freshly generated pairs for 3 epochs:

    python train_dpo.py --checkpoint checkpoints/model_2.pt \
        --data dpo_data.jsonl --epochs 3 --output-name model_2_dpo.pt

Train on several pair files, with conservative-DPO label smoothing:

    python train_dpo.py --checkpoint checkpoints/model_2.pt \
        --data dpo_l1.jsonl --data dpo_l2.jsonl --epochs 3 \
        --beta 0.1 --label-smoothing 0.1
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

import training_common as tc
import wandb_logging

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "model.py"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "checkpoints"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Align a checkpoint with Direct Preference Optimization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="Path to the model module, e.g. models/model.py or models.model",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="SFT checkpoint to start from; also cloned (frozen) as the reference.",
    )
    parser.add_argument(
        "--data",
        required=True,
        action="append",
        metavar="PATH",
        help="Preference-pair JSONL from generate_dpo_data.py. Repeat to combine files.",
    )
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO temperature: higher keeps the policy closer to the reference.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Conservative-DPO (cDPO) smoothing in [0, 0.5); assumes this fraction "
        "of preference labels are flipped/noisy.",
    )
    parser.add_argument(
        "--length-normalize",
        action="store_true",
        help="Divide each chain's log-prob by its response length before the loss "
        "(counters the bias toward shorter chains).",
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Peak learning rate.")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Final (cosine) LR.")
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1.")
    parser.add_argument("--beta2", type=float, default=0.95, help="AdamW beta2.")
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the best DPO checkpoint is saved.",
    )
    parser.add_argument(
        "--output-name",
        default="best_dpo_model.pt",
        help="Filename for the saved checkpoint inside --output-dir.",
    )
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, or cuda.")
    wandb_logging.add_wandb_args(parser, artifacts=True)
    return parser


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


class PreferenceDataset(Dataset):
    """Tokenized (prompt + chosen) / (prompt + rejected) sequences.

    Each item is (chosen_seq, rejected_seq, prompt_len). The prompt is encoded
    once with a leading <bos>; each completion is encoded with a trailing <eos>,
    reproducing the SFT surface form. prompt_len is the number of prompt tokens
    (shared by both sequences) so the trainer can mask the prompt out of the loss.
    """

    def __init__(self, records, tokenizer, max_seq_len):
        self.samples = []
        skipped = 0
        for record in records:
            prompt_ids = tokenizer.encode(
                record["prompt"], add_bos=True, add_eos=False
            )
            chosen_ids = tokenizer.encode(
                record["chosen"], add_bos=False, add_eos=True
            )
            rejected_ids = tokenizer.encode(
                record["rejected"], add_bos=False, add_eos=True
            )
            chosen_seq = torch.cat([prompt_ids, chosen_ids])
            rejected_seq = torch.cat([prompt_ids, rejected_ids])

            # A pair is only usable if both sequences fit and each has at least
            # one response token to score.
            if chosen_ids.numel() < 1 or rejected_ids.numel() < 1:
                skipped += 1
                continue
            if max(chosen_seq.numel(), rejected_seq.numel()) > max_seq_len:
                skipped += 1
                continue

            self.samples.append((chosen_seq, rejected_seq, prompt_ids.numel()))
        self.skipped = skipped

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def _pad_and_mask(seqs, prompt_lens, pad_id):
    """Right-pad sequences and build the response-only target mask.

    Returns (inputs, targets, mask) where inputs = seq[:-1], targets = seq[1:],
    and mask[b, i] is True exactly when target i is a real response token (past
    the prompt and before the padding). Padding lands at the end, so with causal
    attention it never affects the scored positions.
    """
    max_len = max(seq.size(0) for seq in seqs)
    batch_size = len(seqs)
    padded = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    real_lens = torch.empty(batch_size, dtype=torch.long)
    for index, seq in enumerate(seqs):
        padded[index, : seq.size(0)] = seq
        real_lens[index] = seq.size(0)

    inputs = padded[:, :-1]
    targets = padded[:, 1:]
    # Target position i predicts seq[i + 1], so compare against seq index i + 1.
    seq_pos = torch.arange(1, max_len).unsqueeze(0)  # [1, max_len - 1]
    prompt_lens = torch.as_tensor(prompt_lens).unsqueeze(1)
    real_lens = real_lens.unsqueeze(1)
    mask = (seq_pos >= prompt_lens) & (seq_pos < real_lens)
    return inputs, targets, mask


def make_collate(pad_id):
    def collate(batch):
        chosen_seqs = [item[0] for item in batch]
        rejected_seqs = [item[1] for item in batch]
        prompt_lens = [item[2] for item in batch]
        return {
            "chosen": _pad_and_mask(chosen_seqs, prompt_lens, pad_id),
            "rejected": _pad_and_mask(rejected_seqs, prompt_lens, pad_id),
        }

    return collate


def load_pairs(data_paths):
    records = []
    for path_str in data_paths:
        path = tc.resolve_path(path_str)
        if not path.exists():
            raise SystemExit(f"Preference-pair file not found: {path}")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    if not records:
        raise SystemExit("No preference pairs found in the provided --data files.")
    return records


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------


def load_policy_and_reference(model_arg, checkpoint_arg, device):
    """Build two models from the same checkpoint: a trainable policy and a frozen reference."""
    checkpoint_path = tc.resolve_path(checkpoint_arg)
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    module = tc.load_model_module(model_arg)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise SystemExit(f"'model_state_dict' key not found in checkpoint: {checkpoint_path}")
    state = checkpoint["model_state_dict"]

    def build_and_load():
        model = tc.build_model(module)
        try:
            model.load_state_dict(state)
        except RuntimeError as exc:
            raise SystemExit(
                f"Could not load '{checkpoint_path}' into the architecture from "
                f"'{model_arg}'. The checkpoint was trained with different "
                f"hyperparameters than models/model.py declares.\n\nOriginal error:\n{exc}"
            ) from exc
        return model.to(device)

    policy = build_and_load()
    reference = build_and_load()
    reference.eval()
    for param in reference.parameters():
        param.requires_grad_(False)
    return policy, reference, checkpoint_path


# --------------------------------------------------------------------------
# DPO loss
# --------------------------------------------------------------------------


def sequence_logprobs(model, inputs, targets, mask, length_normalize):
    """Sum of per-token log-probs over the masked (response) positions.

    Returns a [batch] tensor. With length_normalize, each sum is divided by its
    response-token count so long and short chains are compared on equal footing.
    """
    logits = model(inputs)
    logp = F.log_softmax(logits, dim=-1)
    token_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    masked = token_logp * mask
    summed = masked.sum(dim=-1)
    if length_normalize:
        lengths = mask.sum(dim=-1).clamp(min=1)
        return summed / lengths
    return summed


def _to_device(packed, device):
    inputs, targets, mask = packed
    return inputs.to(device), targets.to(device), mask.to(device)


def dpo_loss(policy, reference, batch, device, beta, label_smoothing, length_normalize):
    """Compute the DPO loss and a dict of monitoring metrics for one batch."""
    chosen = _to_device(batch["chosen"], device)
    rejected = _to_device(batch["rejected"], device)

    policy_chosen = sequence_logprobs(policy, *chosen, length_normalize)
    policy_rejected = sequence_logprobs(policy, *rejected, length_normalize)
    with torch.no_grad():
        ref_chosen = sequence_logprobs(reference, *chosen, length_normalize)
        ref_rejected = sequence_logprobs(reference, *rejected, length_normalize)

    # log( pi(y|x) / ref(y|x) ) for chosen and rejected; the reference terms are
    # the KL leash, and the intractable partition function cancelled in the
    # derivation, leaving this simple difference-of-log-ratios.
    logits = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)

    # Conservative DPO: with label_smoothing = 0 this is the standard
    # -logsigmoid(beta * logits).
    losses = (
        -F.logsigmoid(beta * logits) * (1 - label_smoothing)
        - F.logsigmoid(-beta * logits) * label_smoothing
    )
    loss = losses.mean()

    # Implicit rewards r(x, y) = beta * log( pi / ref ), detached for logging.
    chosen_reward = beta * (policy_chosen - ref_chosen).detach()
    rejected_reward = beta * (policy_rejected - ref_rejected).detach()
    metrics = {
        "loss": loss.item(),
        "reward_accuracy": (chosen_reward > rejected_reward).float().mean().item(),
        "reward_margin": (chosen_reward - rejected_reward).mean().item(),
        "chosen_reward": chosen_reward.mean().item(),
        "rejected_reward": rejected_reward.mean().item(),
    }
    return loss, metrics


# --------------------------------------------------------------------------
# Train / evaluate
# --------------------------------------------------------------------------


@torch.no_grad()
def evaluate(policy, reference, loader, device, args):
    policy.eval()
    totals = {}
    count = 0
    for batch in loader:
        _, metrics = dpo_loss(
            policy, reference, batch, device,
            args.beta, args.label_smoothing, args.length_normalize,
        )
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
    policy.train()
    return {key: value / max(count, 1) for key, value in totals.items()}


def save_checkpoint(path, model, epoch, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "vocab_size": model.vocab_size,
            "max_seq_len": model.max_seq_len,
            "dpo_metrics": metrics,
        },
        path,
    )


def main():
    parser = parse_args()
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = tc.select_device(args.device)
    policy, reference, checkpoint_path = load_policy_and_reference(
        args.model, args.checkpoint, device
    )

    records = load_pairs(args.data)
    dataset = PreferenceDataset(records, policy.tokenizer, policy.max_seq_len)
    if len(dataset) < 2:
        raise SystemExit(
            f"Only {len(dataset)} usable pairs (skipped {dataset.skipped}); need >= 2."
        )

    test_size = max(1, int(len(dataset) * args.test_split))
    train_size = len(dataset) - test_size
    generator = torch.Generator().manual_seed(args.seed)
    train_set, test_set = random_split(dataset, [train_size, test_size], generator=generator)

    pad_id = policy.tokenizer.eos_token_id
    collate = make_collate(pad_id)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate
    )

    optimizer = tc.configure_optimizer(
        policy, lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2)
    )
    total_steps = args.epochs * max(len(train_loader), 1)
    scheduler = tc.build_scheduler(
        optimizer, warmup_steps=args.warmup_steps, total_steps=total_steps,
        peak_lr=args.lr, min_lr=args.min_lr,
    )

    run = wandb_logging.init_run(
        args,
        config={
            "method": "dpo",
            "checkpoint": str(checkpoint_path),
            "device": str(device),
            "beta": args.beta,
            "label_smoothing": args.label_smoothing,
            "length_normalize": args.length_normalize,
            "pairs_total": len(dataset),
            "pairs_skipped": dataset.skipped,
            "train_pairs": train_size,
            "test_pairs": test_size,
            "total_steps": total_steps,
            "lr": args.lr,
        },
        job_type="dpo",
    )

    best_checkpoint = args.output_dir / args.output_name
    print(f"Device: {device}")
    print(f"Reference/start checkpoint: {checkpoint_path}")
    print(f"Usable pairs: {len(dataset)} (skipped {dataset.skipped}) | "
          f"train: {train_size} | test: {test_size}")
    print(f"beta: {args.beta} | label smoothing: {args.label_smoothing} | "
          f"length-normalize: {args.length_normalize}")
    print(f"peak lr: {args.lr:.2e} -> {args.min_lr:.2e} | warmup: {args.warmup_steps} | "
          f"total steps: {total_steps}")
    print(f"Saving best checkpoint to: {best_checkpoint}")
    if run is not None:
        print(f"Tracking run on W&B: {wandb_logging.run_location(run)}")

    global_step = 0
    best = {"loss": float("inf"), "epoch": 0}
    policy.train()

    try:
        for epoch in range(1, args.epochs + 1):
            running = {}
            batches = 0
            for batch in train_loader:
                loss, metrics = dpo_loss(
                    policy, reference, batch, device,
                    args.beta, args.label_smoothing, args.length_normalize,
                )
                optimizer.zero_grad()
                loss.backward()
                grad_norm = None
                if args.grad_clip and args.grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        policy.parameters(), args.grad_clip
                    )
                optimizer.step()
                scheduler.step()
                global_step += 1
                batches += 1

                for key, value in metrics.items():
                    running[key] = running.get(key, 0.0) + value

                if run is not None:
                    log = {f"dpo/{key}": value for key, value in metrics.items()}
                    log["dpo/lr"] = scheduler.get_last_lr()[0]
                    if grad_norm is not None:
                        log["dpo/grad_norm"] = grad_norm.item()
                    wandb_logging.log(run, log, step=global_step)

            train_avg = {key: value / max(batches, 1) for key, value in running.items()}
            eval_avg = evaluate(policy, reference, test_loader, device, args)
            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"train loss: {train_avg['loss']:.4f} "
                f"(acc {train_avg['reward_accuracy']*100:.1f}%) | "
                f"test loss: {eval_avg['loss']:.4f} "
                f"(acc {eval_avg['reward_accuracy']*100:.1f}%, "
                f"margin {eval_avg['reward_margin']:.3f})"
            )

            if run is not None:
                wandb_logging.log(
                    run,
                    {
                        "epoch": epoch,
                        **{f"test/{key}": value for key, value in eval_avg.items()},
                        "train/loss_epoch": train_avg["loss"],
                    },
                    step=global_step,
                )

            if eval_avg["loss"] < best["loss"]:
                best.update(loss=eval_avg["loss"], epoch=epoch)
                save_checkpoint(best_checkpoint, policy, epoch, eval_avg)
                print(f"  -> New best model saved (test loss: {eval_avg['loss']:.4f})")

        print(f"\nTraining complete. Best test loss: {best['loss']:.4f} "
              f"(epoch {best['epoch']})")
        print(f"Best checkpoint: {best_checkpoint}")
        wandb_logging.summary(
            run, {"best_test_loss": best["loss"], "best_epoch": best["epoch"]}
        )
        if getattr(args, "wandb_artifacts", False) and best_checkpoint.exists():
            wandb_logging.log_checkpoint(
                run, best_checkpoint,
                metadata={"test_loss": best["loss"], "epoch": best["epoch"]},
            )
    finally:
        wandb_logging.finish(run)


if __name__ == "__main__":
    main()
