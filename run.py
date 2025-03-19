from collections import OrderedDict
import chess_gameplay as chg

import torch
import yaml
from model import Model

def fix_checkpoint():
  # Load the original checkpoint
  checkpoint = torch.load('checkpoint.pt', map_location='cpu')

  # If the checkpoint has a "model" key, adjust that state_dict:
  if "model" in checkpoint:
      orig_state_dict = checkpoint["model"]
      new_state_dict = OrderedDict()
      for key, value in orig_state_dict.items():
          new_key = key.replace("module.", "")
          new_state_dict[new_key] = value
      checkpoint["model"] = new_state_dict
  else:
      # Otherwise, adjust the top-level dictionary
      new_state_dict = OrderedDict()
      for key, value in checkpoint.items():
          new_key = key.replace("module.", "")
          new_state_dict[new_key] = value
      checkpoint = new_state_dict

  # Save the adjusted checkpoint
  torch.save(checkpoint, 'adjusted_checkpoint.pt')

  # Later, load the model weights:
  adjusted_checkpoint = torch.load('adjusted_checkpoint.pt', map_location='cpu')
  model.load_state_dict(adjusted_checkpoint["model"])

# All necessary arguments for your model to initialize with must be saved in a YAML file called "model_config.yaml"
# so that your model can be instantiated exactly as follows. Your model must NOT require any initialization arguments
# besides those described in your "model_config.yaml" file.

model_config = yaml.safe_load(open("model_config.yaml"))
model = Model(**model_config)

# adjusted checkpoint loading
# fix_checkpoint()
# checkpoint = torch.load("adjusted_checkpoint.pt", map_location=torch.device('cpu'))
# checkpoint loading
checkpoint = torch.load("checkpoint.pt", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint["model"])

# Your model checkpoint must be called "checkpoint.pt" and must be a dictionary-like object with your model weights
# stored at the key "model" so that it can be loaded into your model exactly as follows.

# checkpoint = torch.load("checkpoint.pt", map_location="cpu")
# model.load_state_dict(checkpoint["model"])

# Note: when you load your model weights you may see the following warning. You can safely ignore this warning.

ignore = """
/root/.chess/lib/python3.10/site-packages/torch/cuda/__init__.py:619: UserWarning: Can't initialize NVML
  warnings.warn("Can't initialize NVML")
"""

# Regular agents
# agents = {'white': chg.Agent(), 'black': chg.Agent()}

# Agents with torch models
agents = {"white": chg.Agent(model), "black": chg.Agent(model)}

teams = {"white": "Team White", "black": "Team Black"}
game_result = chg.play_game(
    agents,
    teams,
    max_moves=50,
    min_seconds_per_move=0,
    verbose=True,
    poseval=True,
    image_path="demo.png",
)