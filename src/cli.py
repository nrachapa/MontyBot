#!/usr/bin/env python3
import contextlib
import os
import random

import chess
import click
import numpy as np
import torch

from src.config import CONFIG
from src.infra.s3_manager import S3Manager
from src.training.game import MoveMapper, encode_board
from src.training.system import Factory


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def cli(verbose):
    """MontyBot Chess Engine CLI"""
    if verbose:
        click.echo("Verbose mode enabled")


@cli.command()
@click.option("--iterations", type=int, help="Training iterations")
@click.option("--filters", type=int, help="Model filters")
@click.option("--blocks", type=int, help="Model blocks")
@click.option("--batch-size", type=int, help="Batch size")
@click.option("--cloud/--local", default=False, help="Cloud vs local training")
def train(iterations, filters, blocks, batch_size, cloud):
    """Launch model training"""
    # Override CONFIG
    if iterations:
        CONFIG.iterations = iterations
    if filters:
        CONFIG.filters = filters
    if blocks:
        CONFIG.blocks = blocks
    if batch_size:
        CONFIG.batch_size = batch_size

    if cloud:
        click.echo("🚀 Launching cloud training...")
        from src.infra.sagemaker_manager import launch_training

        launch_training()
    else:
        click.echo("🏠 Starting local training...")
        from src.training.trainer import Trainer

        trainer = Trainer()
        trainer.train(CONFIG.iterations)


@cli.command()
@click.option("--games", type=int, help="Number of evaluation games")
@click.option("--threshold", type=float, help="Win rate threshold")
@click.option("--is-local", is_flag=True, help="Skip S3 download and use local model")
def evaluate(games, threshold, is_local):
    """Evaluate trained model against random opponent"""
    if games:
        CONFIG.eval_games = games
    if threshold:
        CONFIG.eval_threshold = threshold

    click.echo("🎲 Starting evaluation...")
    _evaluate_impl(is_local)


def _evaluate_impl(is_local=False):
    """Implementation moved from old evaluate() function"""

    s3_manager = S3Manager(bucket=CONFIG.bucket, prefix=CONFIG.prefix)

    checkpoint = None

    if is_local:
        # Local mode: only check for local model.pt
        if os.path.exists("model.pt"):
            click.echo("📂 Loading local model.pt...")
            checkpoint = torch.load("model.pt", map_location=torch.device("cpu"))
        else:
            click.echo("❌ No local model.pt found. Run local training first or omit --is-local flag")
            return
    else:
        # Cloud mode: download from S3 (ignore local model.pt)
        click.echo("☁️ Downloading from S3...")
        checkpoint = s3_manager.download_checkpoint()

        if checkpoint is None:
            click.echo("❌ No trained model found in S3. Run cloud training first")
            return

    # Use config from checkpoint if available to match model architecture
    if "config" in checkpoint:
        saved_config = checkpoint["config"]
        CONFIG.filters = saved_config.get("filters", CONFIG.filters)
        CONFIG.blocks = saved_config.get("blocks", CONFIG.blocks)

    network = Factory.create_network("alphazero")
    network.load_state_dict(checkpoint["model_state"])
    model = network.eval()
    device_str = CONFIG.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    games = CONFIG.eval_games
    threshold = CONFIG.eval_threshold

    wins, draws, losses = 0, 0, 0

    for i in range(games):
        board = chess.Board()
        model_white = i % 2 == 0

        while not board.is_game_over() and len(board.move_stack) < 200:
            if (board.turn == chess.WHITE) == model_white:
                state = torch.from_numpy(encode_board(board)).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits, _ = model(state)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                legal_moves = list(board.legal_moves)
                legal_indices = [MoveMapper.to_index(m) for m in legal_moves]
                move = legal_moves[np.argmax(probs[legal_indices])]
            else:
                move = random.choice(list(board.legal_moves))
            board.push(move)

        result = board.result()
        if result == "1/2-1/2":
            draws += 1
        elif (result == "1-0") == model_white:
            wins += 1
        else:
            losses += 1

    win_rate = wins / (wins + draws + losses)
    click.echo(f"Win Rate: {win_rate:.1%} ({wins}-{draws}-{losses})")
    if win_rate > threshold:
        click.echo("✅ Model OK!")
    else:
        click.echo(f"⚠️ Need training (< {threshold:.1%})")


@cli.group()
def config():
    """Configuration management"""
    pass


@config.command("show")
def config_show():
    """Show current configuration"""
    click.echo("📋 Current Configuration:")
    click.echo("  Model Architecture:")
    click.echo(f"    Filters: {CONFIG.filters}")
    click.echo(f"    Blocks: {CONFIG.blocks}")
    click.echo(f"    Input Planes: {CONFIG.input_planes}")
    click.echo(f"    Action Size: {CONFIG.action_size}")
    click.echo("  Training:")
    click.echo(f"    Learning Rate: {CONFIG.learning_rate}")
    click.echo(f"    Batch Size: {CONFIG.batch_size}")
    click.echo(f"    Iterations: {CONFIG.iterations}")
    click.echo("  Runtime:")
    click.echo(f"    Device: {CONFIG.device}")


@config.command("validate")
def config_validate():
    """Validate current configuration"""
    try:
        # Pydantic validates on creation, so check current state
        CONFIG.model_validate(CONFIG.model_dump())
        click.echo("✅ Configuration is valid")
    except Exception:
        click.echo("❌ Configuration is invalid")
        raise click.ClickException("Invalid configuration")


@config.command("reset")
@click.confirmation_option(prompt="Reset all configuration to defaults?")
def config_reset():
    """Reset configuration to defaults"""
    from src.config import MontyBotConfig

    global CONFIG
    CONFIG = MontyBotConfig()
    click.echo("✅ Configuration reset to defaults")


if __name__ == "__main__":
    import multiprocessing

    with contextlib.suppress(RuntimeError):
        multiprocessing.set_start_method("spawn", force=True)
    cli()
