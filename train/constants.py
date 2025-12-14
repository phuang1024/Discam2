import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Model params.
# Frame step of sequential input frames.
MODEL_FRAME_STEP = 5
# Number of past frames in input.
MODEL_NUM_FRAMES = 5
MODEL_INPUT_RES = (640, 360)
