"""Training configuration presets for AlphaGomoku"""

# Quick test configuration
TEST_CONFIG = {
    'epochs': 5,
    'selfplay_games': 10,
    'batch_size': 128,
    'lr': 0.01,
    'num_simulations': 100,
    'model_blocks': 4,
    'model_channels': 32,
}

# Development configuration
DEV_CONFIG = {
    'epochs': 50,
    'selfplay_games': 100,
    'batch_size': 256,
    'lr': 0.005,
    'num_simulations': 400,
    'model_blocks': 8,
    'model_channels': 48,
}

# Production configuration for strong model
PROD_CONFIG = {
    'epochs': 200,
    'selfplay_games': 500,
    'batch_size': 512,
    'lr': 0.001,
    'num_simulations': 800,
    'model_blocks': 12,
    'model_channels': 64,
}

# High-end configuration for maximum strength
STRONG_CONFIG = {
    'epochs': 400,
    'selfplay_games': 1000,
    'batch_size': 1024,
    'lr': 0.0005,
    'num_simulations': 1600,
    'model_blocks': 16,
    'model_channels': 128,
}