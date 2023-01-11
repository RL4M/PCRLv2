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

    model_name = "simance_multi_crop_luna_pretask_1.0_240.pt"
    weight_path = Path(str(args.base_weight_path), model_name)

    # Note: the weights are pickled!!!!
    with open(weight_path, "rb") as f:
        weights = pickle.load(file)
    model_dict = torch.load(weights, map_location=torch.device("cpu"))["state_dict"]
    model.load_state_dict(torch.load(model_dict))
    pylogger.debug("model loaded successfully")
    # pylogger.debug(f"model dict= \n {model_dict}\n")
    # model.load_state_dict(torch.load(model_dict)) #error here.
    # Need to unpickle first, then load into pytorch
    # see: https://stackoverflow.com/questions/13939913/how-to-test-if-a-file-has-been-created-by-pickle/73523239#73523239?newreg=292d1a1f979548b987e04a987c07f8e8
    # DEBUG : __main__ : opcodes[0].name:PROTO
    # DEBUG : __main__ : opcodes[-1].name:STOP
    # the results of the
    # with open(weight_path, "rb") as f:
    #     pickle = f.read()
    #     output = pickletools.genops(pickle)
    #     opcodes = []
    #     for opcode in output:
    #         opcodes.append(opcode[0])
    #     pylogger.debug(f"opcodes[0].name:{opcodes[0].name}")
    #     pylogger.debug(f"opcodes[-1].name:{opcodes[-1].name}")
    # model_dict = torch.load(weight_path, map_location=torch.device('cpu'))["state_dict"]
