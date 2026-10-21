"""Real Part 1 training run per diffusion.toml (seed 2026, 25000 updates)."""
from src.train import train

train("artifacts/run1", seed=2026, updates=25000, batch_size=512,
      checkpoint_updates=(0, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 25000),
      loss_every=1)
print("TRAINING_DONE")
