from data.datamodule import MeshDataModule
from model.module import MeshGraphNet

from lightning.pytorch.cli import LightningCLI, LightningArgumentParser

import warnings
warnings.filterwarnings("ignore")

class MyLightningCLI(LightningCLI):
    """Custom Lightning CLI to define default arguments."""
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        default_callbacks = [
            
        ]

        logger = {
            "class_path": "lightning.pytorch.loggers.TensorBoardLogger",
            "init_args": {
                "save_dir": "/scratch-big/upelissier/80-Tests/meshgraphnets/",
                "name": "logs/"
            },
        }

        parser.set_defaults(
            {
                "data.data_dir": "/scratch-big/upelissier/80-Tests/meshgraphnets/data/",
                "data.dataset_name": "cylinder_flow",
                "data.field": "velocity",
                "data.time_steps": 600,
                "data.idx_lim_train": 48,
                "data.idx_lim_val": 1,
                "data.idx_lim_test": 1,
                "data.time_step_lim": 100,
                "data.batch_size_train": 1,
                "data.batch_size_valid": 1,
                "data.batch_size_test": 1,
                "data.input_dim_node": 14,
                "data.input_dim_edge": 3,
                "data.output_dim": 2,

                "model.path": "/scratch-big/upelissier/80-Tests/meshgraphnets/",
                "model.dataset": "/scratch-big/upelissier/80-Tests/meshgraphnets/data/",
                "model.logs": "/scratch-big/upelissier/80-Tests/meshgraphnets/logs/",
                "model.noise_std": 0,
                "model.num_layers": 10,
                "model.input_dim_node": 14,
                "model.input_dim_edge": 3,
                "model.hidden_dim": 64,
                "model.output_dim": 2,
                "model.optimizer": "torch.optim.AdamW",

                "trainer.max_epochs": 100,
                "trainer.accelerator": "gpu",
                "trainer.devices": 1,
                "trainer.check_val_every_n_epoch": 1,
                "trainer.log_every_n_steps": 1,
                "trainer.logger": logger,
                "trainer.callbacks": default_callbacks,
            },
        )

if __name__ == '__main__':
    """https://medium.com/stanford-cs224w/learning-mesh-based-flow-simulations-on-graph-networks-44983679cf2d"""
    cli = MyLightningCLI(
        model_class=MeshGraphNet,
        datamodule_class=MeshDataModule,
        seed_everything_default=42,
    )