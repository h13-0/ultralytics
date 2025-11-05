import math
from copy import deepcopy
from pathlib import Path

import torch
from torch.nn import functional as F

from ultralytics.data import dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn.tasks import YOLOTVPModel
from ultralytics.utils import LOGGER, TQDM
from ultralytics.utils.torch_utils import smart_inference_mode, select_device


class YOLOTVPValidator(DetectionValidator):
    """
    A mixin class for YOLOE model validation that handles both text and visual prompt embeddings.

    This mixin provides functionality to validate YOLOE models using either text or visual prompt embeddings.
    It includes methods for extracting visual prompt embeddings from samples, preprocessing batches, and
    running validation with different prompt types.

    Attributes:
        device (torch.device): The device on which validation is performed.
        args (namespace): Configuration arguments for validation.
        dataloader (DataLoader): DataLoader for validation data.
    """

    @staticmethod
    def _crop_region(image_tensor, bbox_tensor):
        """
        Crop a region from ``image_tensor`` using a normalized ``bbox_tensor``.

        Args:
            image_tensor (torch.Tensor): Image tensor of shape (3, H, W) on CPU.
            bbox_tensor (torch.Tensor): Normalized xywh tensor on CPU.

        Returns:
            torch.Tensor | None: Cropped tensor or None if the crop is invalid.
        """
        _, height, width = image_tensor.shape
        cx, cy, bw, bh = bbox_tensor.tolist()
        x1 = max(int(math.floor((cx - bw / 2) * width)), 0)
        y1 = max(int(math.floor((cy - bh / 2) * height)), 0)
        x2 = min(int(math.ceil((cx + bw / 2) * width)), width)
        y2 = min(int(math.ceil((cy + bh / 2) * height)), height)

        if x1 >= width or y1 >= height:
            return None
        if x2 <= x1:
            x2 = min(x1 + 1, width)
        if y2 <= y1:
            y2 = min(y1 + 1, height)

        crop = image_tensor[:, y1:y2, x1:x2]
        if crop.numel() == 0:
            return None
        return crop.clone()

    def preprocess(self, batch):
        """Preprocess batch data, ensuring visuals are on the same device as images."""
        batch = super().preprocess(batch)
        if "visuals" in batch:
            batch["visuals"] = batch["visuals"].to(batch["img"].device)

        imgs = batch["img"]
        bboxes = batch.get("bboxes")
        batch_indices = batch.get("batch_idx")

        num_boxes = bboxes.shape[0]

        if num_boxes == 0:
            batch["crops"] = []
            return batch

        image_cache = {}
        bboxes_cpu = bboxes.cpu()
        batch_indices_cpu = batch_indices.long().cpu()

        crops = []
        for idx in range(num_boxes):
            img_idx = int(batch_indices_cpu[idx].item())
            if img_idx not in image_cache:
                image_cache[img_idx] = imgs[img_idx].detach().cpu()
            crop = self._crop_region(image_cache[img_idx], bboxes_cpu[idx])
            if crop is None:
                continue
            crops.append(crop)
        batch["crops"] = crops

        return batch

    @smart_inference_mode()
    def get_visual_pe(self, dataloader, model):
        """
        Extract visual prompt embeddings from training samples.

        This function processes a dataloader to compute visual prompt embeddings for each class
        using a YOLOE model. It normalizes the embeddings and handles cases where no samples
        exist for a class.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader providing training samples.
            model (YOLOTVPModel): The YOLOTVP model from which to extract visual prompt embeddings.

        Returns:
            (torch.Tensor): Visual prompt embeddings with shape (1, num_classes, embed_dim).
        """
        assert isinstance(model, YOLOTVPModel)
        names = [name.split("/", 1)[0] for name in list(dataloader.dataset.data["names"].values())]
        visual_pe = torch.zeros(len(names), model.model[-1].embed, device=self.device)

        cache_path = (
                Path(dataloader.dataset.img_path[0]).parent /
                f"validset_visual_embeddings_{model.variant.replace(':', '_').replace('/', '_')}.pt")
        if cache_path.exists():
            LOGGER.info(f"Reading existed cache from '{cache_path}'")
            visual_pe = torch.load(cache_path).to(self.device)
            return visual_pe

        cls_visual_num = torch.zeros(len(names))

        desc = "Get visual prompt embeddings from samples"

        # Count samples per class
        for batch in dataloader:
            cls = batch["cls"].squeeze(-1).to(torch.int).unique()
            count = torch.bincount(cls, minlength=len(names))
            cls_visual_num += count

        cls_visual_num = cls_visual_num.to(self.device)

        # Extract visual prompt embeddings
        pbar = TQDM(dataloader, total=len(dataloader), desc=desc)
        for batch in pbar:
            batch = self.preprocess(batch)
            preds = model.get_visual_pe(batch["crops"])  # (B, max_n, embed_dim)

            batch_idx = batch["batch_idx"]
            for i in range(preds.shape[0]):
                cls = batch["cls"][batch_idx == i].squeeze(-1).to(torch.int).unique(sorted=True)
                pad_cls = torch.ones(preds.shape[1], device=self.device) * -1
                pad_cls[: cls.shape[0]] = cls
                for c in cls:
                    visual_pe[c] += preds[i][pad_cls == c].sum(0) / cls_visual_num[c]

        # Normalize embeddings for classes with samples, set others to zero
        visual_pe[cls_visual_num != 0] = F.normalize(visual_pe[cls_visual_num != 0], dim=-1, p=2)
        visual_pe[cls_visual_num == 0] = 0
        visual_pe = visual_pe.unsqueeze(0)
        torch.save(visual_pe, cache_path)
        return visual_pe


    @smart_inference_mode()
    def __call__(self, trainer=None, model=None, refer_data=None, load_vp=False):
        """
        Run validation on the model using either text or visual prompt embeddings.

        This method validates the model using either text prompts or visual prompts, depending
        on the `load_vp` flag. It supports validation during training (using a trainer object)
        or standalone validation with a provided model.

        Args:
            trainer (object, optional): Trainer object containing the model and device.
            model (YOLOEModel, optional): Model to validate. Required if `trainer` is not provided.
            refer_data (str, optional): Path to reference data for visual prompts.
            load_vp (bool): Whether to load visual prompts. If False, text prompts are used.

        Returns:
            (dict): Validation statistics containing metrics computed during validation.
        """
        if trainer is not None:
            self.device = trainer.device
            model = trainer.ema.ema
            names = [name.split("/", 1)[0] for name in list(self.dataloader.dataset.data["names"].values())]

            if load_vp:
                LOGGER.info("Validate using the visual prompt.")
                self.args.half = False
                # Directly use the same dataloader for visual embeddings extracted during training
                vpe = self.get_visual_pe(self.dataloader, model)
                model.set_classes(names, tpe=None, vpe=vpe)
            else:
                LOGGER.info("Validate using the text prompt.")
                tpe = model.get_text_pe(names)
                model.set_classes(names, tpe=tpe, vpe=None)
            stats = super().__call__(trainer, model)
        else:
            if refer_data is not None:
                assert load_vp, "Refer data is only used for visual prompt validation."
            self.device = select_device(self.args.device)

            if isinstance(model, str):
                from ultralytics.nn.tasks import attempt_load_weights

                model = attempt_load_weights(model, device=self.device, inplace=True)
            model.eval().to(self.device)
            data = check_det_dataset(refer_data or self.args.data)
            names = [name.split("/", 1)[0] for name in list(data["names"].values())]

            if load_vp:
                LOGGER.info("Validate using the visual prompt.")
                self.args.half = False
                # TODO: need to check if the names from refer data is consistent with the evaluated dataset
                # could use same dataset or refer to extract visual prompt embeddings
                dataloader = self.get_vpe_dataloader(data)
                vpe = model.get_visual_pe(dataloader, model)
                model.set_classes(names, vpe)
                stats = super().__call__(model=deepcopy(model))
            else:
                LOGGER.info("Validate using the text prompt.")
                tpe = model.get_text_pe(names)
                model.set_classes(names, tpe)
                stats = super().__call__(model=deepcopy(model))
        return stats
