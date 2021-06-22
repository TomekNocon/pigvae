import torch
import pytorch_lightning as pl
from graphae.modules import GraphAE


class PLGraphAE(pl.LightningModule):

    def __init__(self, hparams, critic):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.graph_ae = GraphAE(hparams)
        self.critic = critic
        self.tau_scheduler = TauScheduler(hparams)

    def forward(self, graph, training, tau):
        graph_pred, perm, mu, logvar = self.graph_ae(graph, training, tau)
        return graph_pred, perm, mu, logvar

    def training_step(self, graph, batch_idx):
        tau = self.tau_scheduler.tau
        self.log("tau", tau)
        graph_pred, perm, mu, logvar = self(
            graph=graph,
            training=True,
            tau=tau
        )
        loss = self.critic(
            graph_true=graph,
            graph_pred=graph_pred,
            perm=perm,
            mu=mu,
            logvar=logvar,
        )
        self.log_dict(loss)
        return loss

    def validation_step(self, graph, batch_idx):
        tau = self.tau_scheduler.tau
        graph_pred, perm, mu, logvar = self(
            graph=graph,
            training=True,
            tau=tau
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
            tau=tau
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
        self.log_dict(metrics)
        self.log_dict(metrics_soft)

    def on_validation_epoch_end(self) -> None:
        self.tau_scheduler()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.graph_ae.parameters(), lr=self.hparams["lr"], betas=(0.9, 0.98))
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=0.99,
        )
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

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure=None,
                       second_order_closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        # warm up lr
        if self.trainer.global_step < 10000:
            lr_scale = min(1., float(self.trainer.global_step + 1) / 10000.)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.lr

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()


class TauScheduler(object):
    def __init__(self, hparams):
        self.tau = hparams["tau"]
        self.factor = hparams["tau_decay_factor"]
        self.step_size = hparams["tau_decay_step_size"]
        self.steps = 0

    def __call__(self):
        self.steps += 1
        if self.step_size > 0:
            if self.steps >= self.step_size:
                self.tau *= self.factor
                self.steps = 0
