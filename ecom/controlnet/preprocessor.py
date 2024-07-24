import gc
import os
import PIL.Image
import torch
from controlnet_aux import (
    CannyDetector,
    ContentShuffleDetector,
    HEDdetector,
    LineartAnimeDetector,
    LineartDetector,
    MidasDetector,
    MLSDdetector,
    NormalBaeDetector,
    OpenposeDetector,
    PidiNetDetector,
)


class Preprocessor:
    MODEL_ID = "/tmp/pycharm_project_823/weights/annotators_aux/"

    def __init__(self):
        self.model = None
        self.name = ""

    def load(self, name: str) -> None:
        if name == self.name:
            return
        if name == "HED":
            self.model = HEDdetector.from_pretrained(self.MODEL_ID)
        elif name == "Midas":
            self.model = MidasDetector.from_pretrained(self.MODEL_ID)
        elif name == "MLSD":
            self.model = MLSDdetector.from_pretrained(self.MODEL_ID)
        elif name == "Openpose":
            self.model = OpenposeDetector.from_pretrained(self.MODEL_ID)
        elif name == "PidiNet":
            self.model = PidiNetDetector.from_pretrained(self.MODEL_ID)
        elif name == "NormalBae":
            self.model = NormalBaeDetector.from_pretrained(self.MODEL_ID)
        elif name == "Lineart":
            self.model = LineartDetector.from_pretrained(self.MODEL_ID)
        elif name == "LineartAnime":
            self.model = LineartAnimeDetector.from_pretrained(self.MODEL_ID)
        elif name == "Canny":
            self.model = CannyDetector()
        elif name == "ContentShuffle":
            self.model = ContentShuffleDetector()
        else:
            raise ValueError
        torch.cuda.empty_cache()
        gc.collect()
        self.name = name

    def __call__(self, image, **kwargs) -> PIL.Image.Image:
        return self.model(image, **kwargs)
