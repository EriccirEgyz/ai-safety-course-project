"""
Quick test script to verify beta_schedule implementation
Tests all three modes: fixed, margin_easy, margin_hard
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from trades import trades_loss

# Create a simple test model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32*32*3, 10)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

def test_beta_schedule():
    print("="*60)
    print("Testing beta_schedule implementation")
    print("="*60)

    # Create dummy data
    batch_size = 8
    x = torch.randn(batch_size, 3, 32, 32).cuda()
    y = torch.randint(0, 10, (batch_size,)).cuda()

    # Create model and optimizer
    model = SimpleModel().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Test parameters
    test_configs = [
        {'beta_schedule': 'fixed', 'name': 'Fixed (Original TRADES)'},
        {'beta_schedule': 'margin_easy', 'name': 'Margin Easy (Current)'},
        {'beta_schedule': 'margin_hard', 'name': 'Margin Hard (Reversed)'},
    ]

    for config in test_configs:
        print(f"\n{'='*60}")
        print(f"Testing: {config['name']}")
        print(f"beta_schedule = '{config['beta_schedule']}'")
        print(f"{'='*60}")

        try:
            loss, stats = trades_loss(
                model=model,
                x_natural=x,
                y=y,
                optimizer=optimizer,
                step_size=0.007,
                epsilon=0.031,
                perturb_steps=3,  # Reduced for quick test
                beta=6.0,
                beta_schedule=config['beta_schedule'],
                margin_tau=0.3,
                margin_temperature=0.15,
                beta_i_min=1.0,
                beta_i_max=10.0,
                return_stats=True
            )

            print(f"✓ Loss computed successfully: {loss.item():.4f}")
            print(f"  - Natural loss: {stats['loss_natural']:.4f}")
            print(f"  - Robust loss: {stats['loss_robust']:.4f}")
            print(f"  - Beta_i stats:")
            print(f"    Mean: {stats['beta_i_mean']:.4f}, Std: {stats['beta_i_std']:.4f}")
            print(f"    Min: {stats['beta_i_min']:.4f}, Max: {stats['beta_i_max']:.4f}")

            if config['beta_schedule'] != 'fixed':
                print(f"  - Margin stats:")
                print(f"    Mean: {stats['margin_mean']:.4f}, Std: {stats['margin_std']:.4f}")
                print(f"    p10: {stats['margin_p10']:.4f}, p50: {stats['margin_p50']:.4f}, p90: {stats['margin_p90']:.4f}")
            else:
                print(f"  - Margin stats: N/A (fixed schedule)")

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return False

    print(f"\n{'='*60}")
    print("✓ All tests passed!")
    print("="*60)
    return True

if __name__ == '__main__':
    if torch.cuda.is_available():
        success = test_beta_schedule()
        if not success:
            exit(1)
    else:
        print("CUDA not available, skipping test")
