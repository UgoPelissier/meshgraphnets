import copy
from typing import Optional, List, Tuple, Union
import os
import os.path as osp
import json
import numpy as np

from utils.stats import normalize, unnormalize, load_stats
from utils.save import convert_to_meshio_vtu, vtu_to_xdmf
from data.dataset import NodeType
from model.processor import ProcessorLayer

import torch
from torch.nn import Linear, Sequential, LayerNorm, ReLU
from torch.optim import Optimizer

from torch_geometric.data import Data

import lightning.pytorch as pl
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable


class MeshGraphNet(pl.LightningModule):
    """Lightning module for the MeshNet model."""
    def __init__(
            self,
            path: str,
            dataset: str,
            logs: str,
            noise_std: float,
            num_layers: int,
            input_dim_node: int,
            input_dim_edge: int,
            hidden_dim: int,
            output_dim: int,
            optimizer: OptimizerCallable,
            lr_scheduler: Optional[LRSchedulerCallable] = None
        ) -> None:
        super().__init__()

        self.path = path
        self.dataset = dataset
        self.logs = logs
        self.noise_std = noise_std
        self.num_layers = num_layers

        # encoder convert raw inputs into latent embeddings
        self.node_encoder = Sequential(Linear(input_dim_node, hidden_dim),
                                       ReLU(),
                                       Linear(hidden_dim, hidden_dim),
                                    #    ReLU(),
                                    #    Linear(hidden_dim, hidden_dim),
                                       LayerNorm(hidden_dim))

        self.edge_encoder = Sequential(Linear(input_dim_edge, hidden_dim),
                                       ReLU(),
                                       Linear(hidden_dim, hidden_dim),
                                    #    ReLU(),
                                    #    Linear(hidden_dim, hidden_dim),
                                       LayerNorm(hidden_dim))

        self.processor = torch.nn.ModuleList()
        assert (self.num_layers >= 1), 'Number of message passing layers is not >=1'

        processor_layer=self.build_processor_model()
        for _ in range(self.num_layers):
            self.processor.append(processor_layer(hidden_dim,hidden_dim))


        # decoder: only for node embeddings
        self.decoder = Sequential(Linear(hidden_dim, hidden_dim),
                                  ReLU(),
                                #   Linear(hidden_dim, hidden_dim),
                                #   ReLU(),
                                  Linear(hidden_dim, output_dim))

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        
        self.val_step_outputs = []
        self.val_step_targets = []
        self.current_val_trajectory = 0
        self.last_val_prediction = None

        # For one trajectory vizualization
        self.trajectory_to_save: list[Data] = []

        with open(osp.join(self.dataset, 'raw', 'meta.json'), 'r') as fp:
            meta = json.loads(fp.read())
        self.dt = meta['dt']
        
    def build_processor_model(self):
        return ProcessorLayer

    def forward(
        self,
        batch: Data,
        mean_x: torch.Tensor,
        std_x: torch.Tensor,
        mean_edge: torch.Tensor,
        std_edge: torch.Tensor
    ):
        """
        Encoder encodes graph (node/edge features) into latent vectors (node/edge embeddings)
        The return of processor is fed into the processor for generating new feature vectors
        """ 
        x, edge_index, edge_attr = batch.x, batch.edge_index.long(), batch.edge_attr
        x[:,:2] += self.v_noise(batch, self.noise_std)

        # step 0: normalize
        x, edge_attr = normalize(
            data=[x, edge_attr],
            mean=[mean_x, mean_edge],
            std=[std_x, std_edge]
        )

        # step 1: encode node/edge features into latent node/edge embeddings
        x = self.node_encoder(x) # output shape is the specified hidden dimension

        edge_attr = self.edge_encoder(edge_attr) # output shape is the specified hidden dimension

        # step 2: perform message passing with latent node/edge embeddings
        for i in range(self.num_layers):
            x, edge_attr = self.processor[i](x, edge_index, edge_attr)

        # step 3: decode latent node embeddings into physical quantities of interest
        return self.decoder(x)
    
    def loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        node_type: torch.Tensor,
        mean: torch.Tensor = None,
        std: torch.Tensor = None
    ) -> torch.Tensor:
        """Calculate the loss for the given prediction and target."""
        # get the loss mask for the nodes of the types we calculate loss for
        loss_mask=torch.logical_or((torch.argmax(node_type,dim=1)==torch.tensor(NodeType.NORMAL)),
                                   (torch.argmax(node_type,dim=1)==torch.tensor(NodeType.OUTFLOW)))

        if ((mean is not None) and (std is not None)):
            target_normalized = normalize(
                data=target,
                mean=mean,
                std=std
            )
            # find sum of square errors
            error = torch.sum((pred-target_normalized)**2, dim=1)
        else:
            error = torch.sum((pred-target)**2, dim=1)

        # root and mean the errors for the nodes we calculate loss for
        loss= torch.sqrt(torch.mean(error[loss_mask]))
        
        return loss

    def training_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        """Training step of the model."""
        pred = self(
            batch=batch,
            mean_x=self.mean_vec_x_train,
            std_x=self.std_vec_x_train,
            mean_edge=self.mean_vec_edge_train,
            std_edge=self.std_vec_edge_train
        )
        loss = self.loss(
            pred=pred,
            target=batch.y,
            node_type=batch.x[:,5:],
            mean=self.mean_vec_x_train[:2],
            std=self.std_vec_x_train[:2]
        )
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: Data, batch_idx: int) -> None:
        """Validation step of the model."""
        if self.trainer.sanity_checking:
            self.load_stats()

        # Determine if we need to reset the trajectory
        if batch.traj > self.current_val_trajectory:
            self.current_val_trajectory += 1
            self.last_val_prediction = None

        # Prepare the batch for the current step
        batch = batch.clone()
        if self.last_val_prediction is not None:
            # Update the batch with the last prediction
            batch.x[:,:2] = (self.last_val_prediction.detach())

        if self.current_val_trajectory == 0:
            self.trajectory_to_save.append(batch)

        mask=torch.logical_or(
            (torch.argmax(batch.x[:,5:],dim=1)==torch.tensor(NodeType.NORMAL)),
            (torch.argmax(batch.x[:,5:],dim=1)==torch.tensor(NodeType.OUTFLOW))
        )
        mask = torch.logical_not(mask)
        node_type = batch.x[:,5:]
        target_delta = batch.y

        with torch.no_grad():
            pred_delta_normalized = self(
                batch=batch,
                mean_x=self.mean_vec_x_train,
                std_x=self.std_vec_x_train,
                mean_edge=self.mean_vec_edge_train,
                std_edge=self.std_vec_edge_train
            )

        pred_delta = unnormalize(
            data=pred_delta_normalized,
            mean=self.mean_vec_x_train[:2],
            std=self.std_vec_x_train[:2]
        )

        pred = pred_delta + batch.x[:,:2]
        target = target_delta + batch.x[:,:2]

        # Apply mask to predicted outputs
        pred[mask] = target[mask]
        self.val_step_outputs.append(pred.cpu())
        self.val_step_targets.append(target.cpu())

        self.last_val_prediction = pred

        loss = self.loss(
            pred=pred,
            target=batch.y,
            node_type=node_type
        )
        self.log('valid/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def on_validation_epoch_end(self):
        # Concatenate outputs and targets
        preds = torch.cat(self.val_step_outputs, dim=0)
        targets = torch.cat(self.val_step_targets, dim=0)

        # Compute RMSE over all rollouts
        squared_diff = (preds - targets) ** 2
        all_rollout_rmse = torch.sqrt(squared_diff.mean()).item()

        self.log(
            "val_all_rollout_rmse",
            all_rollout_rmse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # Save trajectory graphs as .vtu files
        save_dir = osp.join("validation", f"epoch_{self.current_epoch}")
        os.makedirs(save_dir, exist_ok=True)
        for idx, graph in enumerate(self.trajectory_to_save):
            try:
                vtu = convert_to_meshio_vtu(graph)
                # Construct filename
                filename = osp.join(save_dir, f"graph_{idx}.vtu")
                # Save the mesh
                vtu.write(filename)
            except Exception as e:
                print(f"Error saving graph {idx} at epoch {self.current_epoch}: {e}")

        # Convert vtu files to XDMF/H5 file
        vtu_files = [osp.join(save_dir, f"graph_{idx}.vtu") for idx in range(len(self.trajectory_to_save))]
        vtu_to_xdmf(osp.join(save_dir, f"graph_{graph.traj.cpu().numpy()[0]}"), vtu_files)

        # Clear stored outputs
        self.val_step_outputs.clear()
        self.val_step_targets.clear()
        self.current_val_trajectory = 0
        self.last_val_prediction = None
        self.trajectory_to_save.clear()

    def test_step(self, batch: Data, batch_idx: int) -> None:
        """Test step of the model."""
        self.load_stats()

    def configure_optimizers(self) -> Union[List[Optimizer], Tuple[List[Optimizer], None]]:
        """Configure the optimizer and the learning rate scheduler."""
        optimizer = self.optimizer(self.parameters())

        if self.lr_scheduler is None:
            return [optimizer]
        else:
            lr_scheduler = self.lr_scheduler(optimizer)
            return [optimizer], [lr_scheduler]
        
    def load_stats(self):
        """Load statistics from the dataset."""
        train_stats, val_stats, test_stats = load_stats(self.dataset, self.device)
        self.mean_vec_x_train, self.std_vec_x_train, self.mean_vec_edge_train, self.std_vec_edge_train, self.mean_vec_y_train, self.std_vec_y_train = train_stats
        self.mean_vec_x_val, self.std_vec_x_val, self.mean_vec_edge_val, self.std_vec_edge_val, self.mean_vec_y_val, self.std_vec_y_val = val_stats
        self.mean_vec_x_test, self.std_vec_x_test, self.mean_vec_edge_test, self.std_vec_edge_test, self.mean_vec_y_test, self.std_vec_y_test = test_stats
    
    def v_noise(self, batch: Data, noise_std: float) -> torch.Tensor:
        """Return noise to add to the velocity field."""
        v_noise = torch.randn_like(batch.x[:,:2])*noise_std
        mask = torch.logical_or(
            (torch.argmax(batch.x[:,5:],dim=1)==torch.tensor(NodeType.NORMAL)),
            (torch.argmax(batch.x[:,5:],dim=1)==torch.tensor(NodeType.OUTFLOW))
        )
        mask = torch.logical_not(mask)
        v_noise[mask] = 0
        return v_noise