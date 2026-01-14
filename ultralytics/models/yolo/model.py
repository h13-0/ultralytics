# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path

from ultralytics.data.augment import LetterBox
from ultralytics.data.build import load_inference_source
from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.models.yolo.yolotvp import YOLOTVPTrainer
from ultralytics.models.yolo.yolotvp.valid import YOLOTVPValidator
from ultralytics.nn.tasks import (
    ClassificationModel,
    DetectionModel,
    OBBModel,
    PoseModel,
    SegmentationModel,
    WorldModel,
    YOLOEModel,
    YOLOESegModel, YOLOTVPModel,
)
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.utils import ARGV, ASSETS, LOGGER, ROOT, YAML


class YOLO(Model):
    """YOLO (You Only Look Once) object detection model."""

    def __init__(self, model="yolo11n.pt", task=None, verbose=False):
        """
        Initialize a YOLO model.

        This constructor initializes a YOLO model, automatically switching to specialized model types
        (YOLOWorld or YOLOE) based on the model filename.

        Args:
            model (str | Path): Model name or path to model file, i.e. 'yolo11n.pt', 'yolo11n.yaml'.
            task (str | None): YOLO task specification, i.e. 'detect', 'segment', 'classify', 'pose', 'obb'.
                Defaults to auto-detection based on model.
            verbose (bool): Display model info on load.

        Examples:
            >>> from ultralytics import YOLO
            >>> model = YOLO("yolo11n.pt")  # load a pretrained YOLOv11n detection model
            >>> model = YOLO("yolo11n-seg.pt")  # load a pretrained YOLO11n segmentation model
        """
        path = Path(model)
        if "-world" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:  # if YOLOWorld PyTorch model
            new_instance = YOLOWorld(path, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        elif "yoloe" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:  # if YOLOE PyTorch model
            new_instance = YOLOE(path, task=task, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        else:
            # Continue with default YOLO initialization
            super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "classify": {
                "model": ClassificationModel,
                "trainer": yolo.classify.ClassificationTrainer,
                "validator": yolo.classify.ClassificationValidator,
                "predictor": yolo.classify.ClassificationPredictor,
            },
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
            "segment": {
                "model": SegmentationModel,
                "trainer": yolo.segment.SegmentationTrainer,
                "validator": yolo.segment.SegmentationValidator,
                "predictor": yolo.segment.SegmentationPredictor,
            },
            "pose": {
                "model": PoseModel,
                "trainer": yolo.pose.PoseTrainer,
                "validator": yolo.pose.PoseValidator,
                "predictor": yolo.pose.PosePredictor,
            },
            "obb": {
                "model": OBBModel,
                "trainer": yolo.obb.OBBTrainer,
                "validator": yolo.obb.OBBValidator,
                "predictor": yolo.obb.OBBPredictor,
            },
        }


class YOLOWorld(Model):
    """YOLO-World object detection model."""

    def __init__(self, model="yolov8s-world.pt", verbose=False) -> None:
        """
        Initialize YOLOv8-World model with a pre-trained model file.

        Loads a YOLOv8-World model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """
        super().__init__(model=model, task="detect", verbose=verbose)

        # Assign default COCO class names when there are no custom names
        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": WorldModel,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
                "trainer": yolo.world.WorldTrainer,
            }
        }

    def set_classes(self, classes):
        """
        Set the model's class names for detection.

        Args:
            classes (list[str]): A list of categories i.e. ["person"].
        """
        self.model.set_classes(classes)
        # Remove background if it's given
        background = " "
        if background in classes:
            classes.remove(background)
        self.model.names = classes

        # Reset method class names
        if self.predictor:
            self.predictor.model.names = classes


class YOLOE(Model):
    """YOLOE object detection and segmentation model."""

    def __init__(self, model="yoloe-11s-seg.pt", task=None, verbose=False) -> None:
        """
        Initialize YOLOE model with a pre-trained model file.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            task (str, optional): Task type for the model. Auto-detected if None.
            verbose (bool): If True, prints additional information during initialization.
        """
        super().__init__(model=model, task=task, verbose=verbose)

        # Assign default COCO class names when there are no custom names
        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": YOLOEModel,
                "validator": yolo.yoloe.YOLOEDetectValidator,
                "predictor": yolo.detect.DetectionPredictor,
                "trainer": yolo.yoloe.YOLOETrainer,
            },
            "segment": {
                "model": YOLOESegModel,
                "validator": yolo.yoloe.YOLOESegValidator,
                "predictor": yolo.segment.SegmentationPredictor,
                "trainer": yolo.yoloe.YOLOESegTrainer,
            },
        }

    def get_text_pe(self, texts):
        """Get text positional embeddings for the given texts."""
        assert isinstance(self.model, YOLOEModel)
        return self.model.get_text_pe(texts)

    def get_visual_pe(self, img, visual):
        """
        Get visual positional embeddings for the given image and visual features.

        This method extracts positional embeddings from visual features based on the input image. It requires
        that the model is an instance of YOLOEModel.

        Args:
            img (torch.Tensor): Input image tensor.
            visual (torch.Tensor): Visual features extracted from the image.

        Returns:
            (torch.Tensor): Visual positional embeddings.

        Examples:
            >>> model = YOLOE("yoloe-11s-seg.pt")
            >>> img = torch.rand(1, 3, 640, 640)
            >>> visual_features = model.model.backbone(img)
            >>> pe = model.get_visual_pe(img, visual_features)
        """
        assert isinstance(self.model, YOLOEModel)
        return self.model.get_visual_pe(img, visual)

    def set_vocab(self, vocab, names):
        """
        Set vocabulary and class names for the YOLOE model.

        This method configures the vocabulary and class names used by the model for text processing and
        classification tasks. The model must be an instance of YOLOEModel.

        Args:
            vocab (list): Vocabulary list containing tokens or words used by the model for text processing.
            names (list): List of class names that the model can detect or classify.

        Raises:
            AssertionError: If the model is not an instance of YOLOEModel.

        Examples:
            >>> model = YOLOE("yoloe-11s-seg.pt")
            >>> model.set_vocab(["person", "car", "dog"], ["person", "car", "dog"])
        """
        assert isinstance(self.model, YOLOEModel)
        self.model.set_vocab(vocab, names=names)

    def get_vocab(self, names):
        """Get vocabulary for the given class names."""
        assert isinstance(self.model, YOLOEModel)
        return self.model.get_vocab(names)

    def set_classes(self, classes, embeddings):
        """
        Set the model's class names and embeddings for detection.

        Args:
            classes (list[str]): A list of categories i.e. ["person"].
            embeddings (torch.Tensor): Embeddings corresponding to the classes.
        """
        assert isinstance(self.model, YOLOEModel)
        self.model.set_classes(classes, embeddings)
        # Verify no background class is present
        assert " " not in classes
        self.model.names = classes

        # Reset method class names
        if self.predictor:
            self.predictor.model.names = classes

    def val(
        self,
        validator=None,
        load_vp=False,
        refer_data=None,
        **kwargs,
    ):
        """
        Validate the model using text or visual prompts.

        Args:
            validator (callable, optional): A callable validator function. If None, a default validator is loaded.
            load_vp (bool): Whether to load visual prompts. If False, text prompts are used.
            refer_data (str, optional): Path to the reference data for visual prompts.
            **kwargs (Any): Additional keyword arguments to override default settings.

        Returns:
            (dict): Validation statistics containing metrics computed during validation.
        """
        custom = {"rect": not load_vp}  # method defaults
        args = {**self.overrides, **custom, **kwargs, "mode": "val"}  # highest priority args on the right

        validator = (validator or self._smart_load("validator"))(args=args, _callbacks=self.callbacks)
        validator(model=self.model, load_vp=load_vp, refer_data=refer_data)
        self.metrics = validator.metrics
        return validator.metrics

    def predict(
        self,
        source=None,
        stream: bool = False,
        visual_prompts: dict = {},
        refer_image=None,
        predictor=None,
        **kwargs,
    ):
        """
        Run prediction on images, videos, directories, streams, etc.

        Args:
            source (str | int | PIL.Image | np.ndarray, optional): Source for prediction. Accepts image paths,
                directory paths, URL/YouTube streams, PIL images, numpy arrays, or webcam indices.
            stream (bool): Whether to stream the prediction results. If True, results are yielded as a
                generator as they are computed.
            visual_prompts (dict): Dictionary containing visual prompts for the model. Must include 'bboxes' and
                'cls' keys when non-empty.
            refer_image (str | PIL.Image | np.ndarray, optional): Reference image for visual prompts.
            predictor (callable, optional): Custom predictor function. If None, a predictor is automatically
                loaded based on the task.
            **kwargs (Any): Additional keyword arguments passed to the predictor.

        Returns:
            (List | generator): List of Results objects or generator of Results objects if stream=True.

        Examples:
            >>> model = YOLOE("yoloe-11s-seg.pt")
            >>> results = model.predict("path/to/image.jpg")
            >>> # With visual prompts
            >>> prompts = {"bboxes": [[10, 20, 100, 200]], "cls": ["person"]}
            >>> results = model.predict("path/to/image.jpg", visual_prompts=prompts)
        """
        if len(visual_prompts):
            assert "bboxes" in visual_prompts and "cls" in visual_prompts, (
                f"Expected 'bboxes' and 'cls' in visual prompts, but got {visual_prompts.keys()}"
            )
            assert len(visual_prompts["bboxes"]) == len(visual_prompts["cls"]), (
                f"Expected equal number of bounding boxes and classes, but got {len(visual_prompts['bboxes'])} and "
                f"{len(visual_prompts['cls'])} respectively"
            )
        self.predictor = (predictor or self._smart_load("predictor"))(
            overrides={
                "task": self.model.task,
                "mode": "predict",
                "save": False,
                "verbose": refer_image is None,
                "batch": 1,
            },
            _callbacks=self.callbacks,
        )

        if len(visual_prompts):
            num_cls = (
                max(len(set(c)) for c in visual_prompts["cls"])
                if isinstance(source, list) and refer_image is None  # means multiple images
                else len(set(visual_prompts["cls"]))
            )
            self.model.model[-1].nc = num_cls
            self.model.names = [f"object{i}" for i in range(num_cls)]
            self.predictor.set_prompts(visual_prompts.copy())

        self.predictor.setup_model(model=self.model)

        if refer_image is None and source is not None:
            dataset = load_inference_source(source)
            if dataset.mode in {"video", "stream"}:
                # NOTE: set the first frame as refer image for videos/streams inference
                refer_image = next(iter(dataset))[1][0]
        if refer_image is not None and len(visual_prompts):
            vpe = self.predictor.get_vpe(refer_image)
            self.model.set_classes(self.model.names, vpe)
            self.task = "segment" if isinstance(self.predictor, yolo.segment.SegmentationPredictor) else "detect"
            self.predictor = None  # reset predictor

        return super().predict(source, stream, **kwargs)

class YOLOTVP(Model):
    """YOLO-World object detection model."""

    def __init__(self, model="yolo11-tvp.yaml", verbose=False) -> None:
        """
        Initialize YOLOv8-World model with a pre-trained model file.

        Loads a YOLOv8-World model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """
        super().__init__(model=model, task="detect", verbose=verbose)

        # Assign default COCO class names when there are no custom names
        if not hasattr(self.model, "names"):
            self.model.names = yaml_load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": YOLOTVPModel,
                "validator": YOLOTVPValidator,
                "predictor": yolo.detect.DetectionPredictor,
                "trainer": YOLOTVPTrainer,
            }
        }

    def set_classes(self, classes):
        """
        Set the model's class names for detection.

        Args:
            classes (list[str]): A list of categories i.e. ["person"].
        """
        self.model.set_classes(classes)
        # Remove background if it's given
        background = " "
        if background in classes:
            classes.remove(background)
        self.model.names = classes

        # Reset method class names
        if self.predictor:
            self.predictor.model.names = classes

    def get_visual_pe(
        self,
        img,
        bboxes=None,
        classes=None,
        bbox_format="xyxy",
        normalized=False,
        imgsz=None,
        return_feats=True,
    ):
        """
        Get visual positional embeddings from reference image and bounding boxes.

        If bboxes is provided, this method crops the regions, encodes them with CLIP to build visual embeddings,
        and also invokes YOLOTVPDetect.get_vft via a forward pass to obtain visual features.

        Args:
            img (torch.Tensor | PIL.Image | np.ndarray): Input image.
            bboxes (list | torch.Tensor, optional): Bounding boxes for visual prompts.
            classes (list[int], optional): Class indices for each bbox. If None, each bbox is its own class.
            bbox_format (str): "xyxy" or "xywh". For "xywh", x,y is top-left in pixels.
            normalized (bool): If True, bbox values are normalized to [0, 1] based on original image size.
            imgsz (int | tuple, optional): Target letterbox size. Defaults to model imgsz or 640.
            return_feats (bool): If True, also return visual features from YOLOTVPDetect.get_vft.

        Returns:
            (torch.Tensor | tuple[torch.Tensor, torch.Tensor]):
                If bboxes is None, returns CLIP visual embeddings.
                If bboxes is provided and return_feats=True, returns (visual_embeds, visual_feats).
        """
        assert isinstance(self.model, YOLOTVPModel)
        if bboxes is None:
            return self.model.get_visual_pe(img, None)

        import numpy as np
        import torch

        if imgsz is None:
            imgsz = (
                getattr(self.model, "args", {}).get("imgsz")
                or self.overrides.get("imgsz")
                or 640
            )
        if isinstance(imgsz, int):
            imgsz = (imgsz, imgsz)

        if hasattr(img, "convert"):  # PIL
            img_np = np.asarray(img.convert("RGB"))
        elif isinstance(img, torch.Tensor):
            img_t = img.detach().cpu()
            if img_t.ndim == 3:
                img_t = img_t.permute(1, 2, 0)
            img_np = img_t.numpy()
            if img_np.dtype != np.uint8:
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        else:
            img_np = img

        stride = int(getattr(self.model, "stride", torch.tensor([32])).max())
        letterbox = LetterBox(new_shape=imgsz, auto=True, stride=stride)
        img_lb = letterbox(image=img_np)
        h0, w0 = img_np.shape[:2]
        r = min(imgsz[0] / h0, imgsz[1] / w0)
        new_unpad = int(round(w0 * r)), int(round(h0 * r))
        dw, dh = imgsz[1] - new_unpad[0], imgsz[0] - new_unpad[1]
        dw, dh = dw % stride, dh % stride
        dw /= 2
        dh /= 2
        r_w, r_h = r, r
        pad_w, pad_h = dw, dh
        lb_h, lb_w = img_lb.shape[:2]

        bboxes = torch.as_tensor(bboxes).float()
        if bboxes.ndim == 1:
            bboxes = bboxes.unsqueeze(0)
        if bboxes.shape[-1] != 4:
            raise ValueError(f"Expected bboxes shape (..., 4), got {bboxes.shape}.")

        if normalized:
            if bbox_format == "xyxy":
                bboxes[:, [0, 2]] *= w0
                bboxes[:, [1, 3]] *= h0
            elif bbox_format == "xywh":
                bboxes[:, 0] *= w0
                bboxes[:, 1] *= h0
                bboxes[:, 2] *= w0
                bboxes[:, 3] *= h0
            else:
                raise ValueError(f"Unsupported bbox_format: {bbox_format}")

        if bbox_format == "xywh":
            x1 = bboxes[:, 0]
            y1 = bboxes[:, 1]
            x2 = bboxes[:, 0] + bboxes[:, 2]
            y2 = bboxes[:, 1] + bboxes[:, 3]
            bboxes = torch.stack([x1, y1, x2, y2], dim=1)
        elif bbox_format != "xyxy":
            raise ValueError(f"Unsupported bbox_format: {bbox_format}")

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * r_w + pad_w
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * r_h + pad_h
        bboxes[:, 0].clamp_(0, lb_w - 1)
        bboxes[:, 2].clamp_(0, lb_w)
        bboxes[:, 1].clamp_(0, lb_h - 1)
        bboxes[:, 3].clamp_(0, lb_h)

        if classes is None:
            classes = list(range(len(bboxes)))
        if len(classes) != len(bboxes):
            raise ValueError("Length of classes must match number of bboxes.")
        num_classes = max(classes) + 1 if classes else 0

        img_t = torch.from_numpy(img_lb).permute(2, 0, 1).float() / 255.0
        crops = []
        crop_classes = []
        for cls_id, (x1, y1, x2, y2) in zip(classes, bboxes.tolist()):
            x1i, y1i = int(x1), int(y1)
            x2i, y2i = max(int(x2), int(x1) + 1), max(int(y2), int(y1) + 1)
            crop = img_t[:, y1i:y2i, x1i:x2i]
            if crop.numel():
                crops.append(crop)
                crop_classes.append(cls_id)

        if not crops:
            raise ValueError("No valid crops could be extracted from the provided bboxes.")

        clip_feats = self.model.get_visual_pe(crops)
        if clip_feats.ndim == 1:
            clip_feats = clip_feats.unsqueeze(0)

        visual_embeds = torch.zeros(num_classes, clip_feats.shape[-1], dtype=clip_feats.dtype)
        counts = torch.zeros(num_classes, dtype=torch.int)
        for feat, cls_id in zip(clip_feats, crop_classes):
            visual_embeds[cls_id] += feat.detach().cpu()
            counts[cls_id] += 1
        nonzero = counts > 0
        visual_embeds[nonzero] /= counts[nonzero].unsqueeze(1)
        visual_embeds[nonzero] = torch.nn.functional.normalize(visual_embeds[nonzero], dim=-1, p=2)
        visual_embeds = visual_embeds.unsqueeze(0)

        device = next(self.model.parameters()).device
        img_t = img_t.unsqueeze(0).to(device=device, dtype=next(self.model.parameters()).dtype)
        visual_mask = torch.zeros((1, num_classes, lb_h, lb_w), device=device, dtype=img_t.dtype)
        for cls_id, (x1, y1, x2, y2) in zip(classes, bboxes.tolist()):
            x1i, y1i = int(x1), int(y1)
            x2i, y2i = max(int(x2), int(x1) + 1), max(int(y2), int(y1) + 1)
            visual_mask[0, cls_id, y1i:y2i, x1i:x2i] = 1.0
        stride_min = int(getattr(self.model, "stride", torch.tensor([32])).min())
        target_h = max(1, int(lb_h // stride_min))
        target_w = max(1, int(lb_w // stride_min))
        if (lb_h, lb_w) != (target_h, target_w):
            visual_mask = torch.nn.functional.interpolate(
                visual_mask, size=(target_h, target_w), mode="nearest"
            )

        visual_feats = None
        if return_feats:
            visual_feats = self.model.predict(img_t, visual_mask=visual_mask, return_vft=True)

        if return_feats:
            return visual_embeds.to(device), visual_feats
        return visual_embeds.to(device)

    def predict(
        self,
        source=None,
        stream: bool = False,
        visual_prompts=None,
        visual_feats=None,
        names=None,
        predictor=None,
        **kwargs,
    ):
        """
        Run prediction with optional visual prompt embeddings.

        Args:
            source (str | int | PIL.Image | np.ndarray, optional): Source for prediction.
            stream (bool): Whether to stream the prediction results.
            visual_prompts (torch.Tensor, optional): Visual prompt embeddings from `get_visual_pe`.
                Accepts shape (N, D), (1, N, D), or (B, N, D).
            visual_feats (torch.Tensor, optional): Visual features to blend with prompts. Defaults to prompts.
            names (list[str] | dict, optional): Class names aligned with the prompts.
            predictor (callable, optional): Custom predictor function.
            **kwargs (Any): Additional keyword arguments passed to the predictor.
        """
        if isinstance(visual_prompts, (tuple, list)) and len(visual_prompts) == 2 and visual_feats is None:
            visual_prompts, visual_feats = visual_prompts

        if visual_prompts is None and visual_feats is None:
            return super().predict(source, stream, predictor=predictor, **kwargs)

        if visual_prompts is None:
            raise ValueError("visual_prompts is required for YOLOTVP visual prompt inference.")

        if not hasattr(visual_prompts, "ndim"):
            raise TypeError(f"Expected visual_prompts to be a tensor-like object, got {type(visual_prompts)}.")

        if visual_prompts.ndim == 2:
            visual_prompts = visual_prompts.unsqueeze(0)
        if visual_prompts.ndim != 3:
            raise ValueError(f"Expected visual_prompts with 3 dimensions, got shape {visual_prompts.shape}.")

        if visual_feats is None:
            visual_feats = visual_prompts
        elif getattr(visual_feats, "ndim", None) == 2:
            visual_feats = visual_feats.unsqueeze(0)
        if getattr(visual_feats, "ndim", None) != 3:
            raise ValueError(f"Expected visual_feats with 3 dimensions, got shape {visual_feats.shape}.")

        if names is None:
            names = getattr(self.model, "names", None)
            if isinstance(names, dict):
                names = list(names.values())
        if not names or len(names) != visual_prompts.shape[1]:
            names = [f"object{i}" for i in range(visual_prompts.shape[1])]

        self.model.set_classes(names, tpe=None, vpe=visual_prompts)

        if source is None:
            source = "https://ultralytics.com/images/boats.jpg" if self.task == "obb" else ASSETS
            LOGGER.warning(f"'source' is missing. Using 'source={source}'.")

        is_cli = (ARGV[0].endswith("yolo") or ARGV[0].endswith("ultralytics")) and any(
            x in ARGV for x in ("predict", "track", "mode=predict", "mode=track")
        )

        custom = {"conf": 0.25, "batch": 1, "save": is_cli, "mode": "predict", "rect": True}  # method defaults
        args = {**self.overrides, **custom, **kwargs}  # highest priority args on the right

        if not self.predictor:
            self.predictor = (predictor or self._smart_load("predictor"))(overrides=args, _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model, verbose=is_cli)
        else:
            self.predictor.args = get_cfg(self.predictor.args, args)
            if "project" in args or "name" in args:
                self.predictor.save_dir = get_save_dir(self.predictor.args)

        return self.predictor(source=source, stream=stream, visual_embeds=visual_prompts, visual_feats=visual_feats)
