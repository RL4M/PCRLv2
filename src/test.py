# import seaborn, logging, matplotlib, pandas, torch, torchvision, torchaudio, cv2, mlflow, torchmetrics, nibabel
from loggers import create_python_logger
from pathlib import Path

import torch, torchvision
from omegaconf import OmegaConf
from models.pcrlv2_model_3d import PCRLv23d

pylogger = create_python_logger(__name__)

if __name__ == "__main__":
    args = OmegaConf.from_cli()
    pylogger.debug(f"{args}")
    model = PCRLv23d()
    model_name = "simance_multi_crop_luna_pretask_1.0_240.pt"
    weight_path = Path(str(args.base_weight_path), model_name)
    model_dict = torch.load(weight_path)["state_dict"]
    pylogger.debug(f"model= \n {model}\n")
    pylogger.debug(f"state dict= \n {state_dict}\n")
    model.load_state_dict(torch.load(model_dict))
    pylogger.debug("model loaded successfully")
    pylogger.debug(f"{model}")
