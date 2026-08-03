import sys
from arch.dataset import MusicDataset
from arch.symph.model import MusicModel3D
from arch.symph.data_tensor import MusicTensorConverter3D
from arch.config import TRAIN_DATA_PATH, VAL_DATA_PATH
from utils.distributed import world_size
from arch import ddp_trainer

def main(
    meta_loss_factor = 0.05,
    harmo_loss_factor = 0.5,
):
    model = MusicModel3D.from_python_config(
        device=None,
        meta_loss_factor=meta_loss_factor,
        harmo_loss_factor=harmo_loss_factor
    )
    converter = MusicTensorConverter3D()
    train_dataset = MusicDataset(TRAIN_DATA_PATH, converter=converter)
    val_dataset = MusicDataset(VAL_DATA_PATH, converter=converter)
    ddp_trainer.main(
        model, train_dataset, val_dataset,
        exp_typ="stage_two_3d",
        ckpt_resume_path=sys.argv[1] if len(sys.argv) > 1 else None,
        grad_accu_steps=(16 // world_size),
    )

if __name__ == "__main__":
    main()
