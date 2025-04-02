import torch.optim as optim
from lightning.pytorch import LightningModule
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryJaccardIndex
from segmentation_models_pytorch import create_model
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss

class LightningSegmentation(LightningModule):
    def __init__(self, model_name, encoder_name, loss_name):
        super().__init__()
        self.save_hyperparameters()
        self.model = create_model(arch=model_name, encoder_name=encoder_name)

        if loss_name == 'dice':
            self.criterion = DiceLoss('binary')
        elif loss_name == 'focal':
            self.criterion = FocalLoss('binary')

        self.train_metrics = MetricCollection(
            {
                "accuracy": BinaryAccuracy(),
                "jaccard": BinaryJaccardIndex()
            },
            prefix="train_"
        )
        self.valid_metrics = self.train_metrics.clone(prefix="valid_")

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.squeeze().long()
        output = self.model(x).squeeze()

        loss = self.criterion(output, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        batch_value = self.train_metrics(output, y)
        self.log_dict(batch_value, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.squeeze().long()
        output = self.model(x).squeeze()

        loss = self.criterion(output, y)
        self.log("valid_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        batch_value = self.valid_metrics(output, y)
        self.log_dict(batch_value, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.1 ** (epoch / 10))
        return [optimizer], [scheduler]