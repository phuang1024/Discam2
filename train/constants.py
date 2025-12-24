import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Model params.
# Frame step of sequential input frames.
MODEL_FRAME_STEP = 5
# Number of past frames in input.
MODEL_NUM_FRAMES = 1#5
# Divisible by 16.
MODEL_INPUT_RES = (512, 288)

## Training params.
LR = 1e-4
BATCH_SIZE = 32
