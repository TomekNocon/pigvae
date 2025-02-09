import os
import logging
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torchvision import transforms, datasets
import torch
from custom_data import SplitPatches, GraphDataModule, SIZE, PATCH_SIZE, GridGraphDataset
from pigvae.trainer import PLGraphAE
from pigvae.synthetic_graphs.custom_hyperparameter import add_arguments
from pigvae.synthetic_graphs.metrics import Critic

logging.getLogger("lightning").setLevel(logging.WARNING)

def main(hparams):
    # Create directories if they do not exist
    run_dir = os.path.join(hparams.save_dir, f"run{hparams.id}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Starting Run {hparams.id}, checkpoints will be saved in {run_dir}")

    # Define checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir,
        filename="{epoch}-{val_loss:.2f}",
        save_last=True,  # Save the last checkpoint
        save_top_k=1,    # Save the best checkpoint based on `monitor`
        monitor="val_loss",  # Metric to monitor
        mode="min"       # Save the checkpoint with the lowest validation loss
    )

    # Define learning rate monitor
    lr_logger = LearningRateMonitor(logging_interval="step")

    # Define Wandb Logger
    wandb_logger = WandbLogger(project=hparams.wb_project_name, log_model=True)
    wandb_logger.experiment.config.update(vars(hparams))

    # Define model
    critic = Critic
    model = PLGraphAE(hparams.__dict__, critic)

    # Define data transformation
    train_transform = transforms.Compose([
        transforms.Resize((SIZE, SIZE)),
        SplitPatches(PATCH_SIZE)
    ])

    # Load MNIST dataset
    mnist = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # Mean and std for MNIST
    ]),
    )

    train_subset_indices = list(range(100))  # Adjust sample size if needed
    val_subset_indices = list(range(100, 110)) 
    train_graph_kwargs_grid = {
        "grid_size": 6,
        "imgs": mnist.data[train_subset_indices].unsqueeze(1),
        "targets": mnist.targets[train_subset_indices],
        "img_transform": train_transform,
        "channels": [0]
    }
    
    val_graph_kwargs_grid = {
        "grid_size": 6,
        "imgs": mnist.data[val_subset_indices].unsqueeze(1),
        "targets": mnist.targets[val_subset_indices],
        "img_transform": train_transform,
        "channels": [0]
    }
    
    # Define data module
    datamodule = GraphDataModule(
        graph_family=hparams.graph_family,
        train_graph_kwargs=train_graph_kwargs_grid,
        val_graph_kwargs=val_graph_kwargs_grid,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        samples_per_epoch=100,
        distributed_sampler=None
    )
    
    
    # Define trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        val_check_interval= 1, # hparams.eval_freq if not hparams.test else 100,
        accelerator="cpu",
        callbacks=[lr_logger, checkpoint_callback],
        precision=hparams.precision,
        max_epochs=hparams.num_epochs,
        log_every_n_steps=5,
        # gradient_clip_val=0.1,
        # terminate_on_nan=True,
        # reload_dataloaders_every_epoch=True, # https://github.com/Lightning-AI/pytorch-lightning/discussions/7372
        # resume_from_checkpoint=hparams.resume_ckpt if hparams.resume_ckpt != "" else None
    )

    # Train the model
    wandb_logger.watch(model)
    trainer.fit(model=model, datamodule=datamodule)

    # Save the best checkpoint path to W&B
    wandb_logger.experiment.log({"best_model_path": checkpoint_callback.best_model_path})
    wandb_logger.experiment.unwatch(model)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()
    main(args)

