import os
import torch.nn.functional as F
import yaml
from utils.segment.plots import plot_images_and_masks
from utils.segment.dataloaders import create_dataloader
from utils.general import init_seeds
import cv2

seed = 2
init_seeds(seed)

with open("data/hyps/hyp.scratch-med-cp20.yaml") as f:
    hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps

save = False
save_dir = "/home/laughing/codes/yolov5-seg/64"

if save and not os.path.exists(save_dir):
    os.mkdir(save_dir)

dataloader, dataset = create_dataloader(
    "/home/laughing/Downloads/coco128-seg/images",
    imgsz=640,
    batch_size=64,
    stride=32,
    augment=True,
    shuffle=True,
    rank=-1,
    hyp=hyp,
    mask_downsample_ratio=1,
    overlap_mask=True,
)
# cv2.namedWindow("mosaic", cv2.WINDOW_NORMAL)

for i, (imgs, targets, paths, _, masks) in enumerate(dataloader):
    # if getattr(dataset, "downsample_ratio", 1) != 1 and masks is not None:
    masks = masks.float()
    # masks = F.interpolate(
    #     masks[None, :],
    #     (320, 320),
    #     mode="bilinear",
    #     align_corners=False,
    # ).squeeze(0)
    # imgs = F.interpolate(
    #     imgs.float(),
    #     (320, 320),
    #     mode="bilinear",
    #     align_corners=False,
    # )

    result = plot_images_and_masks(
        images=imgs,
        targets=targets,
        paths=paths,
        masks=masks,
        # fname=osp.join(save_dir, f"{i}.jpg"),
    )
    # cv2.imshow("mosaic", result[:, :, ::-1])
    # if cv2.waitKey(0) == ord("q"):  # q to quit
    #     break
