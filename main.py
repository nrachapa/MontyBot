#!/usr/bin/env python3
"""Main training script with ONNX export."""

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from training.config import Config
from training.trainer import Trainer
try:
    from training.train import set_seeds
except ImportError:
    def set_seeds(seed: int) -> None:
        import random
        import numpy as np
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

from training.onnx_export import export_to_onnx, validate_onnx_model


def resolve_device(pref: str) -> str:
    """Resolve device preference to actual device."""
    import torch
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if pref == "mps":
        return "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    # auto
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Train and export AlphaZero model to ONNX.")
    p.add_argument("--iterations", type=int, required=True, help="Number of training iterations")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Device to use")
    p.add_argument("--onnx-path", type=str, default="model.onnx", help="Output ONNX file path")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    p.add_argument("--validate", action="store_true", default=True, help="Validate ONNX export")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--checkpoint-every", type=int, default=1000, help="Save checkpoint every N iterations")
    p.add_argument("--final-model", type=str, default="final_model.pt", help="Final model path")
    return p.parse_args()


def main():
    """Main training loop."""
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    log = logging.getLogger("main")

    device = resolve_device(args.device)
    log.info(f"Using device: {device}")

    set_seeds(args.seed)

    # Create config with device override
    cfg = Config()
    cfg.device = device

    trainer = Trainer(cfg)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    metrics = {"iter": [], "loss": []}

    interrupted = {"flag": False}
    def _handle_sigint(sig, frame):
        interrupted["flag"] = True
        log.warning("SIGINT received; will save emergency checkpoint at next safe point.")
    signal.signal(signal.SIGINT, _handle_sigint)

    start = time.perf_counter()
    try:
        for i in range(1, args.iterations + 1):
            # Train one iteration using the actual Trainer API
            results = trainer.train_iteration(i - 1)  # 0-indexed
            loss = results['loss']
            
            metrics["iter"].append(i)
            metrics["loss"].append(float(loss))

            if i % 100 == 0:
                log.info(f"Iter {i}: Loss={loss:.4f}, Buffer={results['buffer_size']}, "
                        f"Memory={results['memory_usage']:.1f}GB")

            if i % args.checkpoint_every == 0:
                path = ckpt_dir / f"checkpoint_iter_{i}.pt"
                trainer.save_checkpoint(i - 1, str(path))
                log.info(f"Saved checkpoint: {path}")

            if interrupted["flag"]:
                path = ckpt_dir / "checkpoint_interrupt.pt"
                trainer.save_checkpoint(i - 1, str(path))
                log.info(f"Saved emergency checkpoint: {path}")
                return

        # Save final model
        import torch
        torch.save(trainer.net.state_dict(), args.final_model)
        log.info(f"Saved final model: {args.final_model}")

        # ONNX export
        export_to_onnx(trainer.net, args.onnx_path, batch_size=1)
        log.info(f"Exported ONNX: {args.onnx_path}")

        # Optional validation
        if args.validate:
            try:
                p_diff, v_diff = validate_onnx_model(args.onnx_path, trainer.net)
                log.info(f"ONNX parity diffs — policy: {p_diff:.3e}, value: {v_diff:.3e}")
            except Exception as e:
                log.warning(f"ONNX validation failed: {e}")

    finally:
        elapsed = time.perf_counter() - start
        log.info(f"Total elapsed: {elapsed:.2f}s")

        # Plot metrics
        if metrics["iter"] and MATPLOTLIB_AVAILABLE:
            try:
                plt.figure(figsize=(10, 6))
                plt.plot(metrics["iter"], metrics["loss"])
                plt.xlabel("Iteration")
                plt.ylabel("Loss")
                plt.title("Training Loss")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig("training_metrics.png", dpi=150)
                plt.close()
                log.info("Saved training_metrics.png")
            except Exception as e:
                log.warning(f"Could not plot metrics: {e}")
        elif not MATPLOTLIB_AVAILABLE:
            log.warning("Matplotlib not available for plotting metrics")


if __name__ == "__main__":
    main()