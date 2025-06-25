from typing import Any, Dict, List, Optional
import torch
import lightning as L
from torchmetrics import JaccardIndex
from torch.nn.modules.loss import _Loss
from minerva.utils.typing import PathLike
from minerva.models.loaders import FromPretrained
from minerva.models.nets.image.sam import Sam
from minerva.models.finetune_adapters import LoRA
from minerva.pipelines.experiment import ModelInstantiator


class SAM_Instantiator(ModelInstantiator):
    def __init__(
        self,
        num_classes: int,
        vit_type: str = "vit-b",
        multimask_output: bool = True,
        loss: Optional[_Loss] = None,
        learning_rate: float = 1e-5,
        optimizer: str = "adam",
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        lr_scheduler: Optional[str] = None,
        lr_scheduler_kwargs: Optional[Dict[str, Any]] = None,
        apply_freeze: Optional[Dict[str, bool]] = None,
        apply_adapter: Optional[Dict[str, LoRA]] = None,
        pixel_mean: Optional[List[float]] = None,
        pixel_std: Optional[List[float]] = None,
        treat_as_binary: bool = True,
        automatic_optimization: bool = True,
    ):
        self.num_classes = num_classes
        self.vit_type = vit_type
        self.multimask_output = multimask_output
        self.loss = loss or torch.nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.apply_freeze = apply_freeze
        self.apply_adapter = apply_adapter
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.treat_as_binary = treat_as_binary
        self.automatic_optimization = automatic_optimization

    def _get_jaccard_metric(self):
        if self.treat_as_binary:
            return JaccardIndex(task="binary")
        else:
            return JaccardIndex(task="multiclass", num_classes=self.num_classes)

    def _create_model(
        self,
        ckpt: Optional[str] = None,
        return_prediction_only: bool = False,
    ) -> L.LightningModule:
        return Sam(
            vit_type=self.vit_type,
            checkpoint=ckpt,
            num_multimask_outputs=self.num_classes,
            iou_head_depth=self.num_classes,
            multimask_output=self.multimask_output,
            apply_freeze=self.apply_freeze,
            apply_adapter=self.apply_adapter,
            pixel_mean=self.pixel_mean,
            pixel_std=self.pixel_std,
            loss_fn=self.loss,
            learning_rate=self.learning_rate,
            optimizer=self.optimizer,
            optimizer_kwargs=self.optimizer_kwargs,
            lr_scheduler=self.lr_scheduler,
            lr_scheduler_kwargs=self.lr_scheduler_kwargs,
            return_prediction_only=return_prediction_only,
            train_metrics={"mIoU": self._get_jaccard_metric()},
            val_metrics={"mIoU": self._get_jaccard_metric()},
            test_metrics={"mIoU": self._get_jaccard_metric()},
            automatic_optimization=self.automatic_optimization,
        )

    def create_model_randomly_initialized(self) -> L.LightningModule:
        return self._create_model(ckpt=None, return_prediction_only=False)

    def create_model_and_load_backbone(
        self, backbone_checkpoint_path: PathLike
    ) -> L.LightningModule:
        return self._create_model(
            ckpt=str(backbone_checkpoint_path), return_prediction_only=False
        )

    def load_model_from_checkpoint(
        self, checkpoint_path: PathLike, return_prediction_only: bool = True
    ) -> L.LightningModule:
        model = self._create_model(
            ckpt=None, return_prediction_only=return_prediction_only
        )

        return FromPretrained(
            model,
            ckpt_path=checkpoint_path,
            strict=False,
        )
