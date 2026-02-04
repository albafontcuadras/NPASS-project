# Model that predicts the NPASS score in the UI.

import random

def predict(features: dict):
    # Demo NPASS-like score between [-1, 1]
    return round(random.uniform(-1, 1), 3)
