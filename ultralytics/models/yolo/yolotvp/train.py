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
from ultralytics.data.augment import LoadVisualPrompt
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
        batch = super().preprocess_batch(batch)

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
        batch = DetectionTrainer.preprocess_batch(self, batch)
        batch["visual_mask"] = batch["visuals"].to(self.device)
        texts = list(itertools.chain(*batch["texts"]))
        visual_map = getattr(self, "visual_embeddings", None) or {}
        eps = 1e-6
        max_samples = 16
        aggregated = []
        for text in texts:
            embed = None
            pool = visual_map.get(text) if isinstance(visual_map, dict) else None
            if isinstance(pool, torch.Tensor) and pool.ndim == 2 and pool.shape[0] > 0:
                num_samples = pool.shape[0]
                sample_count = min(max_samples, num_samples)
                if num_samples == sample_count:
                    sampled = pool[:sample_count]
                else:
                    start = torch.randint(0, num_samples - sample_count + 1, (1,), device=pool.device).item()
                    sampled = pool[start : start + sample_count]
                mean_embed = sampled.mean(dim=0)
                norm = mean_embed.norm(p=2)
                if norm > eps:
                    mean_embed = mean_embed / norm
                embed = mean_embed.to(self.device)
            if embed is None:
                # fallback to text embeddings
                embed = self.text_embeddings[text].to(self.device)
                embed = embed / (embed.norm(p=2) + eps)
            aggregated.append(embed)
        if aggregated:
            emb_tensor = torch.stack(aggregated, dim=0)
            emb_tensor = emb_tensor.reshape(len(batch["texts"]), -1, emb_tensor.shape[-1])
            batch["visual_embeds"] = emb_tensor
        else:
            batch["visual_embeds"] = torch.empty(
                (len(batch["texts"]), 0, next(iter(self.text_embeddings.values())).shape[-1]),
                device=self.device,
            )

        batch_size, nc, embed = batch["visual_embeds"].shape
        pred = self.model.predict(batch["img"], visual_mask=batch["visual_mask"], return_vft=True).cpu()
        visual_feats = torch.zeros(batch_size, nc, embed, device="cpu", dtype=pred.dtype)
        cls_counts = torch.zeros(batch_size, nc, device="cpu", dtype=torch.int)
        for i in range(batch_size):
            inst_mask = (batch["batch_idx"] == i).cpu()
            cls_unique = batch["cls"][inst_mask].squeeze(-1).to(torch.int).unique(sorted=True).cpu()
            if cls_unique.numel() == 0:
                continue
            vfeats = pred[i, : cls_unique.numel()]  # (num_cls_i, 512)
            visual_feats[i].index_add_(0, cls_unique, vfeats)
            cls_counts[i].index_add_(0, cls_unique, torch.ones_like(cls_unique, dtype=cls_counts.dtype))
        nonzero_mask = cls_counts > 0
        # 避免除零：先对非零的类求平均（如需要），再归一化
        visual_feats[nonzero_mask] /= cls_counts[nonzero_mask][..., None]
        visual_feats[nonzero_mask] = torch.nn.functional.normalize(
            visual_feats[nonzero_mask], dim=-1, p=2
        )
        visual_feats[~nonzero_mask] = 0
        batch["visual_feats"] = visual_feats

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
            self.set_text_embeddings(datasets, batch)
            self.set_visual_embeddings(datasets, batch)
        else:
            datasets = [build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=False, multi_modal=True, stride=gs)]
        
        for d in datasets:
            d.transforms.append(LoadVisualPrompt())

        return YOLOConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    def get_batch_crops(self, batch, return_cls=False):
        """
        Extract cropped image patches for all bounding boxes in the batch.
        Args:
            batch (dict): Batch dictionary containing keys 'img', 'bboxes', and 'batch_idx'.
            return_cls (bool): If True, also return class indices aligned with the crops.
        Returns:
            List[torch.Tensor] | Tuple[List[torch.Tensor], List[int]]:
                Cropped image tensors (each in CHW format on CPU), optionally with class indices.
        """
        imgs = batch["img"]
        bboxes = batch.get("bboxes")
        batch_indices = batch.get("batch_idx")
        cls_targets = batch.get("cls") if return_cls else None
        if bboxes is None or batch_indices is None:
            raise KeyError("Batch must contain 'bboxes' and 'batch_idx' to extract crops.")
        bboxes_cpu = bboxes.cpu()
        batch_indices_cpu = batch_indices.long().cpu()
        if return_cls:
            if cls_targets is None:
                raise KeyError("Batch must contain 'cls' to return class indices alongside crops.")
            cls_cpu = cls_targets.view(-1).long().cpu()
        image_cache = {}
        crops = []
        classes = [] if return_cls else None
        iterator = zip(bboxes_cpu, batch_indices_cpu, cls_cpu) if return_cls else zip(bboxes_cpu, batch_indices_cpu)
        for items in iterator:
            if return_cls:
                bbox, img_idx_tensor, cls_idx = items
            else:
                bbox, img_idx_tensor = items
            img_idx = int(img_idx_tensor.item())
            if img_idx not in image_cache:
                image_cache[img_idx] = imgs[img_idx].detach().cpu()
            crop = self._crop_region(image_cache[img_idx], bbox)
            if crop is not None:
                crops.append(crop)
                if return_cls:
                    classes.append(int(cls_idx.item()))
        if return_cls:
            return crops, classes
        return crops

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
            names = getattr(dataset, "names", None)
            if names is None and hasattr(dataset, "data"):
                names = dataset.data.get("names", None)
            cache_path = (
                    Path(dataset.img_path).parent /
                    f"trainset_visual_embeddings_{model.variant.replace(':', '_').replace('/', '_')}.pt")
            visual_map = {}
            cache_loaded = False
            if cache_path.exists():
                LOGGER.info(f"Reading existed cache from '{cache_path}'")
                visual_map = torch.load(cache_path)
                if isinstance(visual_map, dict) and all(isinstance(v, torch.Tensor) and v.ndim == 2 for v in visual_map.values()):
                    cache_loaded = True
                else:
                    LOGGER.warning("Cached visual embeddings are in an unexpected format. Rebuilding cache.")
            if not cache_loaded:
                dataloader = build_dataloader(
                    dataset=dataset,
                    batch=batch_size,
                    workers=4,
                    shuffle=True,
                    rank=-1,
                )
                pbar = TQDM(dataloader, total=len(dataloader), desc=desc)
                for batch in pbar:
                    crops, cls_indices = self.get_batch_crops(batch, return_cls=True)
                    if not crops:
                        continue
                    feats = model.get_visual_pe(crops).squeeze(0)
                    if feats.ndim == 1:
                        feats = feats.unsqueeze(0)
                    for feat, cls_idx in zip(feats, cls_indices):
                        cls_idx = int(cls_idx)
                        if isinstance(names, dict):
                            class_name = names.get(cls_idx, str(cls_idx))
                        elif isinstance(names, (list, tuple)):
                            class_name = names[cls_idx] if 0 <= cls_idx < len(names) else str(cls_idx)
                        else:
                            class_name = str(cls_idx)
                        visual_map.setdefault(class_name, []).append(feat.detach().cpu())
                LOGGER.info(f"Caching text embeddings to '{cache_path}'")
                stacked_map = {k: torch.stack(v, dim=0) for k, v in visual_map.items() if len(v)}
                torch.save(stacked_map, cache_path)
                visual_map = stacked_map
            else:
                visual_map = {k: v.detach().cpu() for k, v in visual_map.items()}
            for class_name, embeds in visual_map.items():
                if class_name in visual_embeddings:
                    visual_embeddings[class_name] = torch.cat((visual_embeddings[class_name], embeds), dim=0)
                else:
                    visual_embeddings[class_name] = embeds
        self.visual_embeddings = visual_embeddings
