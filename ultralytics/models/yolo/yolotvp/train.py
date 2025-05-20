import itertools
from collections import OrderedDict

import torch

from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
from ultralytics.data import build_yolo_dataset, build_grounding, YOLOConcatDataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import YOLOTVPModel
from ultralytics.utils import RANK, DEFAULT_CFG, checks
from ultralytics.utils.torch_utils import de_parallel


def on_pretrain_routine_end(trainer):
    """Callback to set up model classes and text encoder at the end of the pretrain routine."""
    if RANK in {-1, 0}:
        # Set class names for evaluation
        names = [name.split("/")[0] for name in list(trainer.test_loader.dataset.data["names"].values())]
        de_parallel(trainer.ema.ema).set_classes(names, cache_clip_model=False)
    device = next(trainer.model.parameters()).device
    trainer.text_model, _ = trainer.clip.load("ViT-B/32", device=device)
    for p in trainer.text_model.parameters():
        p.requires_grad_(False)

class YOLOTVPTrainer(DetectionTrainer):
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

        # Import and assign clip
        try:
            import clip
        except ImportError:
            checks.check_requirements("git+https://github.com/ultralytics/CLIP.git")
            import clip
        self.clip = clip
        self._text_feats = OrderedDict()
        self._text_feats_num_limit = -1

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Return WorldModel initialized with specified config and weights.

        Args:
            cfg (Dict | str, optional): Model configuration.
            weights (str, optional): Path to pretrained weights.
            verbose (bool): Whether to display model info.

        Returns:
            (WorldModel): Initialized WorldModel.
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
        self.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)

        return model

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO Dataset for training or validation.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`.

        Returns:
            (Dataset): YOLO dataset configured for training or validation.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(
            self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs, multi_modal=mode == "train"
        )

    def get_text_feats(self, texts: list[str], device, dtype) -> torch.Tensor:
        """
        Get and cache the features of text.

        Args:
            texts (list[str]):
            device (torch.device):
            dtype (torch.dtype):
        """
        # Find text features that have not appeared before.
        seen = set()
        new_texts = [
            text for text in texts
            if not (text in seen or seen.add(text))
               and text not in self._text_feats
        ]

        if new_texts:
            new_tokens = self.clip.tokenize(new_texts).to(device)
            new_feats = self.text_model.encode_text(new_tokens).to(dtype=dtype)
            # The new feature will be automatically placed at the end of the dict.
            self._text_feats.update(zip(new_texts, new_feats))

        # Splicing features into results
        features = []
        for text in texts:
            # Move the features used to the end.
            feat = self._text_feats[text]
            self._text_feats.move_to_end(text)
            features.append(feat)

        # Cache management(LRU)
        if self._text_feats_num_limit > 0:
            while len(self._text_feats) > self._text_feats_num_limit:
                # Retrieve the oldest unused features from the dict header.
                oldest_key = next(iter(self._text_feats))
                del self._text_feats[oldest_key]

        return torch.stack(features, dim=0)

    def set_cache_limit(self, num_limit: int):
        """
        Set a limit on the number of text feature caches.

        There is no upper limit when the upper limit of quantity is less than 0(default is -1).

        Args:
            num_limit (int): Cache quantity limit, each cache will occupy about 2KB memory.

        Examples:
        >>> from ultralytics.models.yolo.world import WorldModel
        >>> args = dict(model="yolov8s-world.pt", data="coco8.yaml", epochs=3)
        >>> trainer = WorldTrainer(overrides=args)
        >>> trainer.enable_cache_limit(409600) # Up to approximately 800MB of memory will be occupied.
        >>> trainer.train()
        """
        self._text_feats_num_limit = num_limit

    def preprocess_batch(self, batch):
        """Preprocess a batch of images and text for YOLOWorld training."""
        batch = super().preprocess_batch(batch)

        # Add text features
        texts = list(itertools.chain(*batch["texts"]))
        txt_feats = self.get_text_feats(texts, device=batch["img"].device, dtype=batch["img"].dtype)
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        batch["txt_feats"] = txt_feats.reshape(len(batch["texts"]), -1, txt_feats.shape[-1])
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
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        if mode != "train":
            return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)
        dataset = [
            build_yolo_dataset(self.args, im_path, batch, self.data, stride=gs, multi_modal=True)
            if isinstance(im_path, str)
            else build_grounding(self.args, im_path["img_path"], im_path["json_file"], batch, stride=gs)
            for im_path in img_path
        ]
        return YOLOConcatDataset(dataset) if len(dataset) > 1 else dataset[0]

    def get_dataset(self):
        """
        Get train and validation paths from data dictionary.

        Processes the data configuration to extract paths for training and validation datasets,
        handling both YOLO detection datasets and grounding datasets.

        Returns:
            (str): Train dataset path.
            (str): Validation dataset path.

        Raises:
            AssertionError: If train or validation datasets are not found, or if validation has multiple datasets.
        """
        final_data = {}
        data_yaml = self.args.data
        assert data_yaml.get("train", False), "train dataset not found"  # object365.yaml
        assert data_yaml.get("val", False), "validation dataset not found"  # lvis.yaml
        data = {k: [check_det_dataset(d) for d in v.get("yolo_data", [])] for k, v in data_yaml.items()}
        assert len(data["val"]) == 1, f"Only support validating on 1 dataset for now, but got {len(data['val'])}."
        val_split = "minival" if "lvis" in data["val"][0]["val"] else "val"
        for d in data["val"]:
            if d.get("minival") is None:  # for lvis dataset
                continue
            d["minival"] = str(d["path"] / d["minival"])
        for s in ["train", "val"]:
            final_data[s] = [d["train" if s == "train" else val_split] for d in data[s]]
            # save grounding data if there's one
            grounding_data = data_yaml[s].get("grounding_data")
            if grounding_data is None:
                continue
            grounding_data = grounding_data if isinstance(grounding_data, list) else [grounding_data]
            for g in grounding_data:
                assert isinstance(g, dict), f"Grounding data should be provided in dict format, but got {type(g)}"
            final_data[s] += grounding_data
        # NOTE: to make training work properly, set `nc` and `names`
        final_data["nc"] = data["val"][0]["nc"]
        final_data["names"] = data["val"][0]["names"]
        self.data = final_data
        return final_data["train"], final_data["val"][0]

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

