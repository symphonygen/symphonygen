import sys
from arch.harmo.model import HarmonyGPT
from arch.dataset import MusicDataset
from arch.config import *
from arch.harmo.config import *
from arch.harmo.data_tensor import HarmonyTensorConverter
from arch import ddp_trainer

def main():
    model = HarmonyGPT.from_python_config()
    converter = HarmonyTensorConverter()
    train_dataset = MusicDataset(TRAIN_DATA_PATH, converter=converter)
    val_dataset = MusicDataset(VAL_DATA_PATH, converter=converter)
    ddp_trainer.main(
        model, train_dataset, val_dataset,
        exp_typ="harmony",
        ckpt_resume_path=sys.argv[1] if len(sys.argv) > 1 else None,
        grad_accu_steps=2,
    )

if __name__ == "__main__":
    main()
