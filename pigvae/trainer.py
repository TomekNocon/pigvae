import torch
import pytorch_lightning as pl
from pigvae.modules import GraphAE
from torch.optim.lr_scheduler import LambdaLR 
import math
import wandb

class PLGraphAE(pl.LightningModule):

    def __init__(self, hparams, critic):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.graph_ae = GraphAE(hparams)
        self.critic = critic(hparams)
        self.automatic_optimization = False

    def forward(self, graph, training):
        graph_pred, perm, mu, logvar = self.graph_ae(graph, training, tau=1.0)
        return graph_pred, perm, mu, logvar

    # def training_step(self, graph, batch_idx):
        
    #     opt = self.optimizers()
    #     scheduler = self.lr_schedulers()


    #     graph_pred, perm, mu, logvar = self(
    #         graph=graph,
    #         training=True,
    #     )
    
    #     def closure():
    #         loss = self.critic(
    #             graph_true=graph,
    #             graph_pred=graph_pred,
    #             perm=perm,
    #             mu=mu,
    #             logvar=logvar,
    #             )
    #         opt.zero_grad()
    #         self.manual_backward(loss['loss'])
    #         self.log_dict(loss)
    #         return loss

    #     opt.step(closure=closure)
    #     scheduler.step()
    #     # return loss
    def training_step(self, graph, batch_idx):
        opt = self.optimizers()
        scheduler = self.lr_schedulers()

        graph_pred, perm, mu, logvar = self(
            graph=graph,
            training=True,
        )

        loss_dict = self.critic(
            graph_true=graph,
            graph_pred=graph_pred,
            perm=perm,
            mu=mu,
            logvar=logvar,
        )

        loss = loss_dict["loss"]
        opt.zero_grad()
        self.manual_backward(loss)

        def closure():
            self.log_dict(loss_dict)
            return loss

        opt.optimizer.step(closure=closure)  # Ensure using actual optimizer
        scheduler.step()

        return loss

    def validation_step(self, graph, batch_idx):
        graph_pred, perm, mu, logvar = self(
            graph=graph,
            training=True,
        )
        metrics_soft = self.critic.evaluate(
            graph_true=graph,
            graph_pred=graph_pred,
            perm=perm,
            mu=mu,
            logvar=logvar,
            prefix="val",
        )
        graph_pred, perm, mu, logvar = self(
            graph=graph,
            training=False,
        )
        metrics_hard = self.critic.evaluate(
            graph_true=graph,
            graph_pred=graph_pred,
            perm=perm,
            mu=mu,
            logvar=logvar,
            prefix="val_hard",
        )
        metrics = {**metrics_soft, **metrics_hard}
        self.log_dict(metrics, sync_dist=True, on_epoch=True)
        wandb.log(metrics)
        
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.graph_ae.parameters(),
        lr=self.hparams["lr"], betas=(0.9, 0.98))
        # optimizer = torch.optim.AdamW(
        #     self.graph_ae.parameters(),
        #     lr=self.hparams["lr"], 
        #     betas=(0.9, 0.98),
        #     weight_decay=1e-4  # Adjust weight decay as needed
        # )
        num_training_steps = self.hparams["num_epochs"]
        num_warmup_steps = int(0.01 * num_training_steps)  # 1% of training steps
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                       num_warmup_steps, num_training_steps)
        # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
        # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     optimizer=optimizer,
        #     gamma=0.999,
        # )
        if "eval_freq" in self.hparams:
            scheduler = {
                'scheduler': lr_scheduler,
                'interval': 'step',
                'frequency': 2 * (self.hparams["eval_freq"] + 1)
            }
        else:
            scheduler = {
                'scheduler': lr_scheduler,
                'interval': 'epoch'
            }
        return [optimizer], [scheduler]
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        optimizer = optimizer.optimizer  # Unwrap from LightningOptimizer
        optimizer.step(closure=optimizer_closure)

    
    # def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, closure=None):
    #     # Warm-up learning rate for the first 10,000 steps
    #     if self.trainer.global_step < 10000:
    #         lr_scale = min(1.0, float(self.trainer.global_step + 1) / 10000.0)
    #         for param_group in optimizer.param_groups:
    #             param_group['lr'] = lr_scale * self.hparams.lr

    #     # Update parameters
    #     optimizer.step()
    #     optimizer.zero_grad()


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """
    Creates a schedule with a learning rate that first increases linearly during the warmup period
    and then decreases following a cosine function.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup phase
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)

