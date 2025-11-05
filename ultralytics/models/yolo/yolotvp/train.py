import hashlib
import itertools
import math
from copy import copy
from pathlib import Path

import torch
import torch.nn.functional as F

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.world import WorldTrainer
from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
from ultralytics.data import build_dataloader, YOLOConcatDataset, build_yolo_dataset, build_grounding
from .valid import YOLOTVPValidator
from ultralytics.nn.tasks import YOLOTVPModel
from ultralytics.utils import RANK, DEFAULT_CFG, LOGGER, TQDM
from ultralytics.utils.torch_utils import de_parallel, strip_optimizer


# def on_pretrain_routine_end(trainer):
#     """Callback to set up model classes and text encoder at the end of the pretrain routine."""
#     if RANK in {-1, 0}:
#         # Set class names for evaluation
#         names = [name.split("/")[0] for name in list(trainer.test_loader.dataset.data["names"].values())]
#         de_parallel(trainer.ema.ema).set_classes(names, cache_clip_model=False)

class YOLOTVPTrainer(WorldTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize a YOLOTVPTrainer object with given arguments.

        Args:
            cfg (dict): Configuration for the trainer.
            overrides (dict, optional): Configuration overrides.
            _callbacks (list, optional): List of callback functions.
        """
        if overrides is None:
            overrides = {}
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Return WorldModel initialized with specified config and weights.

        Args:
            cfg (Dict | str, optional): Model configuration.
            weights (str, optional): Path to pretrained weights.
            verbose (bool): Whether to display model info.

        Returns:
            (TVPModel): Initialized TVPModel.
        """
        # NOTE: This `nc` here is the max number of different text samples in one image, rather than the actual `nc`.
        # NOTE: Following the official config, nc hard-coded to 80 for now.
        model = YOLOTVPModel(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=3,
            nc=min(self.data["nc"], 80),
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        # self.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)

        return model

    def generate_text_embeddings(self, texts, batch, cache_dir):
        """
        Generate text embeddings for a list of text samples.

        Args:
            texts (List[str]): List of text samples to encode.
            batch (int): Batch size for processing.
            cache_dir (Path): Directory to save/load cached embeddings.

        Returns:
            (dict): Dictionary mapping text samples to their embeddings.
        """
        if isinstance(self.model, YOLOTVPModel):
            model = self.model
        else:
            model = self.model.module
        cache_path = cache_dir / f"text_embeddings_{model.variant.replace(':', '_').replace('/', '_')}.pt"
        if cache_path.exists():
            LOGGER.info(f"Reading existed cache from '{cache_path}'")
            txt_map = torch.load(cache_path)
            if sorted(txt_map.keys()) == sorted(texts):
                return txt_map
        LOGGER.info(f"Caching text embeddings to '{cache_path}'")
        assert model is not None
        txt_feats = model.get_text_pe(texts, batch, cache_clip_model=False)
        txt_map = dict(zip(texts, txt_feats.squeeze(0)))
        torch.save(txt_map, cache_path)
        return txt_map

    def get_validator(self):
        """Returns a YOLOTVPValidator for YOLO model validation."""
        self.loss_names = "box", "cls", "dfl"
        return YOLOTVPValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def preprocess_batch(self, batch):
        """Preprocess a batch of images and (optional) text prompts for YOLOTVP training."""
        batch = DetectionTrainer.preprocess_batch(self, batch)

        if not getattr(self, "text_embeddings", None):
            return batch
        if "texts" not in batch or not batch["texts"]:
            return batch

        texts = list(itertools.chain(*batch["texts"]))
        txt_feats = torch.stack([self.text_embeddings[text] for text in texts]).to(self.device)
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)

        txt_feats = txt_feats.reshape(len(batch["texts"]), -1, txt_feats.shape[-1])
        batch["txt_feats"] = txt_feats
        batch["embeddings"] = txt_feats
        return batch


class YOLOTVPTrainerFromScratch(YOLOTVPTrainer, WorldTrainerFromScratch):
    """
    A class extending the WorldTrainer for training a world model from scratch on open-set datasets.

    This trainer specializes in handling mixed datasets including both object detection and grounding datasets,
    supporting training YOLO-World models with combined vision-language capabilities.

    Attributes:
        cfg (dict): Configuration dictionary with default parameters for model training.
        overrides (dict): Dictionary of parameter overrides to customize the configuration.
        _callbacks (list): List of callback functions to be executed during different stages of training.

    Examples:
        >>> from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
        >>> from ultralytics import YOLOWorld
        >>> data = dict(
        ...     train=dict(
        ...         yolo_data=["Objects365.yaml"],
        ...         grounding_data=[
        ...             dict(
        ...                 img_path="../datasets/flickr30k/images",
        ...                 json_file="../datasets/flickr30k/final_flickr_separateGT_train.json",
        ...             ),
        ...             dict(
        ...                 img_path="../datasets/GQA/images",
        ...                 json_file="../datasets/GQA/final_mixed_train_no_coco.json",
        ...             ),
        ...         ],
        ...     ),
        ...     val=dict(yolo_data=["lvis.yaml"]),
        ... )
        >>> model = YOLOWorld("yolov8s-worldv2.yaml")
        >>> model.train(data=data, trainer=WorldTrainerFromScratch)
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize a WorldTrainerFromScratch object.

        This initializes a trainer for YOLO-World models from scratch, supporting mixed datasets including both
        object detection and grounding datasets for vision-language capabilities.

        Args:
            cfg (dict): Configuration dictionary with default parameters for model training.
            overrides (dict, optional): Dictionary of parameter overrides to customize the configuration.
            _callbacks (list, optional): List of callback functions to be executed during different stages of training.

        Examples:
            >>> from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
            >>> from ultralytics import YOLOWorld
            >>> data = dict(
            ...     train=dict(
            ...         yolo_data=["Objects365.yaml"],
            ...         grounding_data=[
            ...             dict(
            ...                 img_path="../datasets/flickr30k/images",
            ...                 json_file="../datasets/flickr30k/final_flickr_separateGT_train.json",
            ...             ),
            ...         ],
            ...     ),
            ...     val=dict(yolo_data=["lvis.yaml"]),
            ... )
            >>> model = YOLOWorld("yolov8s-worldv2.yaml")
            >>> model.train(data=data, trainer=WorldTrainerFromScratch)
        """
        if overrides is None:
            overrides = {}
        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO Dataset for training or validation.

        This method constructs appropriate datasets based on the mode and input paths, handling both
        standard YOLO datasets and grounding datasets with different formats.

        Args:
            img_path (List[str] | str): Path to the folder containing images or list of paths.
            mode (str): 'train' mode or 'val' mode, allowing customized augmentations for each mode.
            batch (int, optional): Size of batches, used for rectangular training/validation.

        Returns:
            (YOLOConcatDataset | Dataset): The constructed dataset for training or validation.
        """
        return WorldTrainerFromScratch.build_dataset(self, img_path, mode, batch)

    def plot_training_labels(self):
        """Do not plot labels for YOLO-World training."""
        pass

    def final_eval(self):
        """
        Perform final evaluation and validation for the YOLO-World model.

        Configures the validator with appropriate dataset and split information before running evaluation.

        Returns:
            (dict): Dictionary containing evaluation metrics and results.
        """
        val = self.args.data["val"]["yolo_data"][0]
        self.validator.args.data = val
        self.validator.args.split = "minival" if isinstance(val, str) and "lvis" in val else "val"
        return super().final_eval()


class YOLOTVPVPTrainer(YOLOTVPTrainerFromScratch):
    """Trainer for learning visual prompts alongside text prompts."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize the visual prompt trainer.

        Args:
            cfg (dict): Trainer configuration.
            overrides (dict, optional): Configuration overrides.
            _callbacks (list, optional): Callback functions.
        """
        if overrides is None:
            overrides = {}
        super().__init__(cfg, overrides, _callbacks)
        self.visual_embeddings = None

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

    def preprocess_batch(self, batch):
        """Extend preprocessing to include visual prompt embeddings."""
        batch = super().preprocess_batch(batch)

        imgs = batch["img"]
        bboxes = batch.get("bboxes")
        batch_indices = batch.get("batch_idx")
        cls_targets = batch.get("cls")

        if bboxes is None or batch_indices is None or cls_targets is None:
            raise KeyError("Batch must contain 'bboxes', 'batch_idx', and 'cls' for visual prompt training.")

        model = self.model if isinstance(self.model, YOLOTVPModel) else self.model.module
        head = de_parallel(model).model[-1]
        nc = getattr(head, "base_nc", head.nc)
        embed_dim = getattr(head, "embed_dim", 512)
        batch_size = imgs.shape[0]
        num_boxes = bboxes.shape[0]
        device = self.device

        if num_boxes == 0:
            batch["visual_embeds"] = torch.zeros((0, embed_dim), device=device)
            batch["visual_feats"] = torch.zeros((batch_size, nc, embed_dim), device=device)
            batch["visual_mask"] = torch.zeros((batch_size, nc), dtype=torch.bool, device=device)
            return batch

        # 创建 CPU 版本的标注信息
        bboxes_cpu = bboxes.cpu()
        batch_indices_cpu = batch_indices.long().cpu()
        cls_cpu = cls_targets.view(-1).long().cpu()

        # 限制每张图的最大 bbox 数量，不超过 nc，多余的 bbox 直接忽略
        per_image_counts = torch.bincount(batch_indices_cpu, minlength=batch_size)
        max_bbox = int(per_image_counts.max().item()) if per_image_counts.numel() else 0
        if max_bbox > nc:
            truncated_indices = []
            for img_idx in range(batch_size):
                idxs = (batch_indices_cpu == img_idx).nonzero(as_tuple=False).view(-1)
                if idxs.numel() == 0:
                    continue
                truncated_indices.extend(idxs[:nc].tolist())
            if truncated_indices:
                truncated_indices = torch.tensor(truncated_indices, dtype=torch.long)
                bboxes_cpu = bboxes_cpu[truncated_indices]
                batch_indices_cpu = batch_indices_cpu[truncated_indices]
                cls_cpu = cls_cpu[truncated_indices]
                num_boxes = bboxes_cpu.shape[0]
            else:
                num_boxes = 0

        if num_boxes == 0:
            batch["visual_embeds"] = torch.zeros((0, embed_dim), device=device)
            batch["visual_feats"] = torch.zeros((batch_size, nc, embed_dim), device=device)
            batch["visual_mask"] = torch.zeros((batch_size, nc), dtype=torch.bool, device=device)
            return batch

        image_cache = {}

        dtype = imgs.dtype
        visual_embeds = torch.zeros((num_boxes, embed_dim), device=device, dtype=dtype)
        visual_feats = torch.zeros((batch_size, nc, embed_dim), device=device, dtype=dtype)
        class_counts = torch.zeros((batch_size, nc), device=device, dtype=torch.float32)

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
            encoded = encoded.to(device=device, dtype=imgs.dtype)
            for embed, idx in zip(encoded, valid_indices):
                visual_embeds[idx] = embed

        for idx in range(num_boxes):
            embed = visual_embeds[idx]
            if not torch.any(embed):
                continue
            img_idx = int(batch_indices_cpu[idx].item())
            cls_id = int(cls_cpu[idx].item())
            if cls_id < 0 or cls_id >= nc:
                raise ValueError(f"Class index {cls_id} is out of range for nc={nc}.")
            visual_feats[img_idx, cls_id] += embed
            class_counts[img_idx, cls_id] += 1

        valid_mask = class_counts > 0
        if valid_mask.any():
            counts = class_counts[valid_mask].unsqueeze(-1).to(dtype=visual_feats.dtype)
            visual_feats[valid_mask] /= counts
            visual_feats[valid_mask] = F.normalize(visual_feats[valid_mask], dim=-1, eps=1e-6)

        batch["visual_embeds"] = visual_embeds
        batch["visual_feats"] = visual_feats
        batch["visual_mask"] = valid_mask
        return batch

    def validate(self):
        """Validate using visual prompts extracted during preprocessing."""
        metrics = self.validator(self, load_vp=True)
        if "fitness" in metrics:
            fitness = float(metrics.pop("fitness"))
        elif isinstance(self.loss, torch.Tensor):
            fitness = -float(self.loss.detach().cpu().numpy())
        else:
            fitness = 0.0
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def final_eval(self):
        """Run final evaluation with visual prompts."""
        ckpt = {}
        for f in self.last, self.best:
            if f.exists():
                if f is self.last:
                    ckpt = strip_optimizer(f)
                elif f is self.best:
                    k = "train_results"
                    strip_optimizer(f, updates={k: ckpt[k]} if k in ckpt else None)
                    LOGGER.info(f"\nValidating {f}...")
                    self.validator.args.plots = self.args.plots
                    self.metrics = self.validator(model=f, load_vp=True)
                    self.metrics.pop("fitness", None)
                    self.run_callbacks("on_fit_epoch_end")

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO Dataset for training or validation.

        This method constructs appropriate datasets based on the mode and input paths, handling both
        standard YOLO datasets and grounding datasets with different formats.

        Args:
            img_path (List[str] | str): Path to the folder containing images or list of paths.
            mode (str): 'train' mode or 'val' mode, allowing customized augmentations for each mode.
            batch (int, optional): Size of batches, used for rectangular training/validation.

        Returns:
            (YOLOConcatDataset | Dataset): The constructed dataset for training or validation.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        if mode == "train":
            datasets = [
                build_yolo_dataset(self.args, im_path, batch, self.training_data[im_path], stride=gs, multi_modal=True)
                if isinstance(im_path, str)
                else build_grounding(self.args, im_path["img_path"], im_path["json_file"], batch, stride=gs)
                for im_path in img_path
            ]
            # self.set_visual_embeddings(datasets, batch)
        else:
            datasets = [build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=False, multi_modal=True, stride=gs)]
        return YOLOConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    def get_batch_crops(self, batch):
        """
        Extract cropped image patches for all bounding boxes in the batch.
        Args:
            batch (dict): Batch dictionary containing keys 'img', 'bboxes', and 'batch_idx'.
        Returns:
            List[torch.Tensor]: List of cropped image tensors (each in CHW format on CPU).
        """
        imgs = batch["img"]
        bboxes = batch.get("bboxes")
        batch_indices = batch.get("batch_idx")
        if bboxes is None or batch_indices is None:
            raise KeyError("Batch must contain 'bboxes' and 'batch_idx' to extract crops.")
        bboxes_cpu = bboxes.cpu()
        batch_indices_cpu = batch_indices.long().cpu()
        image_cache = {}
        crops = []
        for bbox, img_idx_tensor in zip(bboxes_cpu, batch_indices_cpu):
            img_idx = int(img_idx_tensor.item())
            if img_idx not in image_cache:
                image_cache[img_idx] = imgs[img_idx].detach().cpu()
            crop = self._crop_region(image_cache[img_idx], bbox)
            if crop is not None:
                crops.append(crop)
        return crops

    def tensor_sha256(self, t, as_hex=True):
        """
        Compute a SHA-256 digest for an arbitrary `torch.Tensor`.

        The tensor is normalized to CPU and C-contiguous layout and hashed via a zero-copy byte view.
        Optionally includes dtype/shape metadata to avoid collisions between identical byte buffers
        interpreted with different views.

        Args:
            t (torch.Tensor): Tensor to hash.
            include_meta (bool, optional): If True, include dtype, shape, and layout metadata in the
                digest to disambiguate views. Defaults to True.
            as_hex (bool, optional): If True, return a hex string; if False, return the raw 32-byte
                digest. Defaults to True.

        Returns:
            str | bytes: The SHA-256 digest in hex (default) or raw bytes.

        Raises:
            TypeError: If `t` is not a `torch.Tensor`.

        Notes:
            - GPU tensors are copied to CPU before hashing.
            - The tensor is detached to avoid holding autograd graph references.

        Example:
            >>> import torch
            >>> d = tensor_sha256(torch.zeros(3, 224, 224, dtype=torch.uint8))
            >>> len(d), isinstance(d, str)
            (64, True)
        """
        if not torch.is_tensor(t):
            raise TypeError(f"tensor_sha256 expects a torch.Tensor, got: {type(t)}")

        # Detach from autograd and normalize device/layout
        x = t.detach()
        if x.device.type != "cpu":
            x = x.cpu()  # GPU -> CPU transfer
        if not x.is_contiguous():
            x = x.contiguous()

        # Share underlying storage with NumPy (contiguous ensured) to avoid extra copies
        arr = x.numpy()  # type: ignore[arg-type]

        h = hashlib.sha256()

        # Zero-copy byte view of tensor data
        h.update(memoryview(arr).cast("B"))

        digest = h.digest()
        return digest.hex() if as_hex else digest

    def set_visual_embeddings(self, datasets, batch_size):
        """
        Set visual embeddings for datasets to accelerate training by caching category names.

        This method collects unique category names from all datasets, then generates and caches text embeddings
        for these categories to improve training efficiency.

        Args:
            datasets (List[Dataset]): List of datasets from which to extract category names.
            batch_size (int | None): Batch size used for processing.

        Notes:
            This method collects category names from datasets that have the 'category_names' attribute,
            then uses the first dataset's image path to determine where to cache the generated text embeddings.
        """
        visual_embeddings = {}
        desc = "Obtain and cache the image embeddings"

        model = self.model if isinstance(self.model, YOLOTVPModel) else self.model.module
        assert model is not None

        for dataset in datasets:
            visual_map = {}
            cache_path = (
                    Path(dataset.img_path).parent /
                    f"trainset_visual_embeddings_{model.variant.replace(':', '_').replace('/', '_')}.pt")
            if cache_path.exists():
                LOGGER.info(f"Reading existed cache from '{cache_path}'")
                visual_map = torch.load(cache_path)
            else:
                dataloader = build_dataloader(
                    dataset=dataset,
                    batch=batch_size,
                    workers=4,
                    shuffle=True,
                    rank=-1,
                )
                pbar = TQDM(dataloader, total=len(dataloader), desc=desc)
                for batch in pbar:
                    crops = self.get_batch_crops(batch)
                    miss_crops = [crop for crop in crops if self.tensor_sha256(crop) not in visual_map]
                    if miss_crops:
                        feats = model.get_visual_pe(miss_crops)
                        crop_hash = [self.tensor_sha256(crop) for crop in miss_crops]
                        visual_map.update(zip(crop_hash, feats.squeeze(0)))

                LOGGER.info(f"Caching text embeddings to '{cache_path}'")
                torch.save(visual_map, cache_path)

            visual_map = {k: v.detach().cpu() for k, v in visual_map.items()}
            visual_embeddings.update(visual_map)

        self.visual_embeddings = visual_embeddings
