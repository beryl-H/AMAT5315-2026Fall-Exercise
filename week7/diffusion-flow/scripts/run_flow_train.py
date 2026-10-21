"""Real Part 3 training run per flow.toml (seed 2029, 25000 updates)."""
from src.flow import train_flow

train_flow("artifacts/flow-run1", seed=2029, updates=25000, batch_size=512,
           checkpoint_updates=(0, 50, 25000), loss_every=1)
print("FLOW_TRAINING_DONE")
