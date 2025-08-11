from typing import List, Dict

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class TrainingMetrics:
    """Track and visualize training metrics."""
    
    def __init__(self):
        self.losses: List[float] = []
        self.win_rates: List[float] = []
        self.iterations: List[int] = []
    
    def add_metrics(self, iteration: int, loss: float, win_rate: float = None):
        """Add training metrics for an iteration."""
        self.iterations.append(iteration)
        self.losses.append(loss)
        if win_rate is not None:
            self.win_rates.append(win_rate)
    
    def plot_metrics(self, save_path: str = 'training_metrics.png'):
        """Plot loss and win rate curves."""
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available. Install with: pip install matplotlib")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.iterations, self.losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        
        if self.win_rates:
            eval_iterations = [self.iterations[i] for i in range(0, len(self.iterations), 1) if i < len(self.win_rates)]
            ax2.plot(eval_iterations, self.win_rates)
            ax2.set_title('Win Rate vs Baseline')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Win Rate')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()