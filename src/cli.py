#!/usr/bin/env python3
import contextlib
import os
import random

import chess
import click
import numpy as np
import torch

from src.config import CONFIG, resolve_device
from src.infra.s3_manager import S3Manager
from src.training.evaluation import SunfishEvaluation
from src.training.game import ChessGame, MoveMapper, Position, encode_board, play_game
from src.training.search import AlphaBetaSearch
from src.training.system import Factory

_HEURISTIC_SEARCH = None  # lazy: the system import pulls the full training graph


def _opponent_policy(state, opponent):
    """Return a move index for the opponent on its turn.

    The caller (play_game) needs an index for apply_move.
    """
    board = state.board
    if opponent == "random":
        move = random.choice(list(board.legal_moves))
        return MoveMapper.to_index(move, board)
    # Heuristic: use the repo's alpha-beta search.
    global _HEURISTIC_SEARCH
    if _HEURISTIC_SEARCH is None:
        _HEURISTIC_SEARCH = AlphaBetaSearch(
            ChessGame(), SunfishEvaluation(), depth=4
        )
    pos = Position(
        board, 1 if board.turn == chess.WHITE else -1,
        board.fen(), encode_board(board)
    )
    policy, _, _ = _HEURISTIC_SEARCH.search(pos)
    legal = list(dict.fromkeys(
        MoveMapper.to_index(m, board) for m in board.legal_moves
    ))
    idx = int(np.argmax(policy[legal]))
    return legal[idx]


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
@click.option("--instance-type", default=None, help="SageMaker instance type (overrides config default)")
@click.option("--on-demand", is_flag=True, help="Use on-demand instances (config default is spot)")
@click.option("--region", default=None, help="AWS region (defaults to $AWS_DEFAULT_REGION or us-east-2)")
def train(iterations, filters, blocks, batch_size, cloud, instance_type, on_demand, region):
    """Launch model training"""
    # --region is the single AWS knob: steer the AWS SDK/boto3 (env) and the
    # S3 bucket name (config) to the same region.
    if region:
        os.environ["AWS_DEFAULT_REGION"] = region
        CONFIG.region = region
        CONFIG.bucket = f"montybot-training-{region}"
    # Override CONFIG
    if iterations:
        CONFIG.iterations = iterations
    if filters:
        CONFIG.filters = filters
    if blocks:
        CONFIG.blocks = blocks
    if batch_size:
        CONFIG.batch_size = batch_size
    if on_demand:
        CONFIG.spot_instances = False

    if cloud:
        click.echo("🚀 Launching cloud training...")
        from src.infra.sagemaker_manager import launch_training

        launch_training(instance_type=instance_type)
    else:
        click.echo("🏠 Starting local training...")
        from src.training.trainer import Trainer

        trainer = Trainer(CONFIG)
        trainer.train(CONFIG.iterations)


@cli.command()
@click.option("--games", type=int, help="Number of evaluation games")
@click.option("--threshold", type=float, help="Win rate threshold")
@click.option("--opponent", type=click.Choice(["random", "heuristic"]),
              default="random", show_default=True,
              help="Opponent: random (legacy) or the repo heuristic (strength bar)")
@click.option("--is-local", is_flag=True, help="Skip S3 download and use local model")
def evaluate(games, threshold, is_local, opponent):
    """Evaluate trained model against an opponent (random or the repo heuristic)"""
    if games:
        CONFIG.eval_games = games
    if threshold:
        CONFIG.eval_threshold = threshold

    click.echo(f"🎲 Starting evaluation (opponent: {opponent})...")
    _evaluate_impl(is_local, opponent)


def _evaluate_impl(is_local=False, opponent: str = "random"):
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

    network = Factory.create_network("alphazero", CONFIG)
    network.load_state_dict(checkpoint["model_state"])
    device = torch.device(resolve_device(CONFIG.device))
    # Move the model to the same device as the input tensors below (which are
    # built on CONFIG.device, e.g. MPS on Apple Silicon); otherwise forward
    # fails with a CPU-weights vs MPS-input type mismatch.
    model = network.to(device).eval()

    games = CONFIG.eval_games
    threshold = CONFIG.eval_threshold

    wins, draws, losses = 0, 0, 0

    # Place model on device once (before the loop) — this is the
    # caller's responsibility; play_game only handles state transitions.

    game = ChessGame()

    for i in range(games):
        model_white = i % 2 == 0

        # Build a policy_fn that routes to model (even moves) or
        # opponent (odd moves) based on whose turn it is.
        def policy(state, _model=model, _device=device, _white=model_white):
            if (state.board.turn == chess.WHITE) == _white:
                # Model's turn: network inference → softmax → argmax
                tensor = torch.from_numpy(state.tensor).unsqueeze(0).to(_device)
                with torch.no_grad():
                    logits, _ = _model(tensor)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                legal_indices = [MoveMapper.to_index(m, state.board)
                                 for m in state.board.legal_moves]
                return int(np.argmax(probs[legal_indices]))
            # Opponent's turn: use opponent policy
            return _opponent_policy(state, opponent)

        # play_game owns the loop: termination + state transitions.
        # Caller manages device placement, policy routing, and
        # win/draw/loss attribution.
        _, winner, _ = play_game(game, policy, max_moves=200)  # (traj, winner, is_capped)

        # Winner is from white's perspective (+1=white wins, -1=black wins, 0=draw).
        # Map to model color: if model plays white, winner==1 → win; else loss.
        if winner == 0:
            draws += 1
        elif (winner == 1) == model_white:
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
    from src.config import CONFIG, MontyBotConfig

    defaults = MontyBotConfig()
    for attr in defaults.model_fields:
        setattr(CONFIG, attr, getattr(defaults, attr))
    click.echo("✅ Configuration reset to defaults")


if __name__ == "__main__":
    import multiprocessing

    with contextlib.suppress(RuntimeError):
        multiprocessing.set_start_method("spawn", force=True)
    cli()
