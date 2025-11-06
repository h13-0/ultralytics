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
from ultralytics.utils.torch_utils import smart_inference_mode, select_device, de_parallel


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
        """Extend preprocessing to include visual prompt embeddings."""
        batch = super().preprocess(batch)
        imgs = batch["img"]
        bboxes = batch.get("bboxes")
        batch_indices = batch.get("batch_idx")
        cls_targets = batch.get("cls")

        if bboxes is None or batch_indices is None or cls_targets is None:
            raise KeyError("Batch must contain 'bboxes', 'batch_idx', and 'cls' for visual prompt training.")

        model = self.model if isinstance(self.model, YOLOTVPModel) else self.model.module
        head = de_parallel(model).model[-1]
        embed_dim = getattr(head, "embed_dim", 512)
        batch_size = imgs.shape[0]
        num_boxes = bboxes.shape[0]
        device = self.device
        dtype = imgs.dtype

        # 创建 CPU 版本的标注信息
        bboxes_cpu = bboxes.cpu()
        batch_indices_cpu = batch_indices.long().cpu()

        per_image_counts = torch.bincount(batch_indices_cpu, minlength=batch_size)
        max_bboxes = int(per_image_counts.max().item()) if per_image_counts.numel() else 0
        head.nc = max_bboxes

        if num_boxes == 0:
            batch["visual_embeds"] = torch.zeros((0, embed_dim), device=device, dtype=dtype)
            batch["visual_feats"] = torch.zeros(
                (batch_size, max_bboxes, embed_dim), device=device, dtype=dtype
            )
            batch["visual_mask"] = torch.zeros((batch_size, max_bboxes), dtype=torch.bool, device=device)
            return batch

        image_cache = {}
        visual_embeds = torch.zeros((num_boxes, embed_dim), device=device, dtype=dtype)
        crops = []
        valid_indices = []
        for idx in range(num_boxes):
            img_idx = int(batch_indices_cpu[idx].item())
            if img_idx not in image_cache:
                image_cache[img_idx] = imgs[img_idx].detach().cpu()
            crop = self._crop_region(image_cache[img_idx], bboxes_cpu[idx])
            if crop is None:
                continue
            crops.append(crop)
            valid_indices.append(idx)

        if crops:
            # encoded = torch.stack([self.visual_embeddings[self.tensor_sha256(crop)] for crop in crops], dim=0)
            encoded = model.get_visual_pe(crops)
            encoded = encoded.to(device=device, dtype=dtype)
            for embed, idx in zip(encoded, valid_indices):
                visual_embeds[idx] = embed

        visual_feats = torch.zeros((batch_size, max_bboxes, embed_dim), device=device, dtype=dtype)
        visual_mask = torch.zeros((batch_size, max_bboxes), dtype=torch.bool, device=device)
        positions = [0] * batch_size
        for idx in range(num_boxes):
            img_idx = int(batch_indices_cpu[idx].item())
            col = positions[img_idx]
            if col >= max_bboxes:
                continue
            visual_feats[img_idx, col] = visual_embeds[idx]
            positions[img_idx] += 1
            visual_mask[img_idx, col] = True

        batch["visual_feats"] = visual_feats
        return batch


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
            else:
                LOGGER.info("Validate using the text prompt.")
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
