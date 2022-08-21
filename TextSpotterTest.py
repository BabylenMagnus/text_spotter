from TESTR.adet.utils.visualizer import TextVisualizer

from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.data.detection_utils import read_image
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T

import os
import sys

import torch
from tqdm import tqdm

from common import setup_cfg, vis_bases
from model import TransformerDetector


class DefaultPredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.

    Compared to using the model directly, this class does the following additions:

    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.

    This is meant for simple demo purposes, so it does the above steps automatically.
    This is not meant for benchmarks or running complicated inference logic.
    If you'd like to do anything more complicated, please refer to its source code as
    examples to build and use the model manually.

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.

    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = TransformerDetector(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions


cfg = setup_cfg(sys.argv[1])

predictor = DefaultPredictor(cfg)
# torch.onnx.export(
#     predictor.model,
#     torch.rand([1, 3, 1488, 1000]),
#     "TESTR_CTW1500.onnx",
#     export_params=True
# )
# print(predictor.model)
# print(type(predictor.model), dir(predictor.model))
# print(predictor)

metadata = MetadataCatalog.get(
    cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
)
print(metadata)
for i in tqdm(sorted(os.listdir(sys.argv[2]))[:50]):
    image = read_image(os.path.join(sys.argv[2], i), format="BGR")
    predictions = predictor(image)
    image = image[:, :, ::-1]

    visualizer = TextVisualizer(image, metadata, instance_mode=ColorMode.IMAGE, cfg=cfg)

    if "bases" in predictions:
        vis_bases(predictions["bases"])
    if "panoptic_seg" in predictions:
        panoptic_seg, segments_info = predictions["panoptic_seg"]
        vis_output = visualizer.draw_panoptic_seg_predictions(
            panoptic_seg.to(torch.device("cpu")), segments_info
        )
    else:
        if "sem_seg" in predictions:
            vis_output = visualizer.draw_sem_seg(
                predictions["sem_seg"].argmax(dim=0).to(torch.device("cpu")))
        if "instances" in predictions:
            instances = predictions["instances"].to(torch.device("cpu"))
            vis_output = visualizer.draw_instance_predictions(predictions=instances)

    vis_output.save(f"{sys.argv[3]}/{i}.jpg")
