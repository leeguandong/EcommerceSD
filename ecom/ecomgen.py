import sys
import torch
from pathlib import Path

root_dir = Path(__file__).resolve().parent
sys.path.append(str(root_dir))
from omegaconf import OmegaConf
from ecom import engineer_log as eng

config_path = str(root_dir / 'configs/config_ecommerce.yaml')
if not Path(config_path).exists():
    raise FileExistsError(f'{config_path} does not exist!')
ecommerce_config = OmegaConf.load(config_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

try:
    ecommerce_sd_model = ""
except Exception as ee:
    eng.log.error(str(ee))
    raise eng.EngParameter(eng.return_pattern(10100, "controlnet初始化异常,error:" + str(ee)))


def run_ecom():
    pass
