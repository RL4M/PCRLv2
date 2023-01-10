# import seaborn, logging, matplotlib, pandas, torch, torchvision, torchaudio, cv2, mlflow, torchmetrics, nibabel
from loggers import create_python_logger
from pathlib import Path
import pickletools
import torch, torchvision
from omegaconf import OmegaConf
from models.pcrlv2_model_3d import PCRLv23d

pylogger = create_python_logger(__name__)

if __name__ == "__main__":
    args = OmegaConf.from_cli()
    pylogger.debug(f"{args}")
    model = PCRLv23d()
    pylogger.debug(f"model= \n {model}\n")
    model_name = "simance_multi_crop_luna_pretask_1.0_240.pt"
    weight_path = Path(str(args.base_weight_path), model_name)
    with open(weight_path, "rb") as f:
        pickle = f.read()
        output = pickletools.genops(pickle)
        opcodes = []
        for opcode in output:
            opcodes.append(opcode[0])
        pylogger.debug(f"opcodes[0].name:{opcodes[0].name}")
        pylogger.debug(f"opcodes[-1].name:{opcodes[-1].name}")
    # model_dict = torch.load(weight_path, map_location=torch.device('cpu'))["state_dict"]

    # # pylogger.debug(f"model dict= \n {model_dict}\n")
    # model.load_state_dict(torch.load(model_dict)) #error here.
