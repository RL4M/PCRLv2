# import seaborn, logging, matplotlib, pandas, torch, torchvision, torchaudio, cv2, mlflow, torchmetrics, nibabel
from loggers import create_python_logger
from pathlib import Path
import pickletools, pickle
import torch, torchvision
from omegaconf import OmegaConf
from models.pcrlv2_model_3d import PCRLv23d

pylogger = create_python_logger(__name__)

if __name__ == "__main__":
    args = OmegaConf.from_cli()
    pylogger.debug(f"{args}")
    model = PCRLv23d()
    pylogger.debug(f"model= \n {model}\n")
    for model_name in [
        "simance_multi_crop_luna_pretask_1.0_240.pth",
        "simance_multi_crop_chexpter_pretask_1.0_240.pth",
        "simance_multi_crop_chest_pretask_1.0_240.pth",
    ]:
        model_dict_path = Path(str(args.base_weight_path), model_name)
        model_dict = torch.load(model_dict_path, map_location=torch.device("cpu"))
        model.load_state_dict(model_dict)
        pylogger.debug(f"model:{model_name} loaded successfully")
