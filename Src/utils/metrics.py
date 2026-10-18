import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import math
from ultralytics.utils import nms
from ultralytics.utils.ops import xywh2xyxy
import numpy as np

def visualize_predictions_range(model, data_module, start_idx, end_idx, conf_thresh=0.25, iou_thresh=0.50):
    """
    Show images from a start index to an end index (for example 10 to 20)
    and draw predictions in red and ground truth in green.
    """
    model.eval()
    device = next(model.parameters()).device

    # Make sure the dataset is loaded
    if getattr(data_module, "val_dataset", None) is None:
        data_module.setup()

    val_dataset = data_module.val_dataset
    total_val_images = len(val_dataset)

    # Clamp indices so they stay within the dataset size
    start_idx = max(0, start_idx)
    end_idx = min(total_val_images, end_idx)
    num_images = end_idx - start_idx

    if num_images <= 0:
        print("❌ Error: Invalid index range or no images left.")
        return

    # Load only the requested images
    images = []
    targets = []
    for i in range(start_idx, end_idx):
        img, tgt = val_dataset[i]
        images.append(img)
        targets.append(tgt)

    images_tensor = torch.stack(images).to(device)

    # Model predictions
    with torch.no_grad():
        outputs = model(images_tensor)
        if isinstance(outputs, tuple): outputs = outputs[0]
        preds = nms.non_max_suppression(outputs, conf_thres=conf_thresh, iou_thres=iou_thresh, nc=1)

    # Layout images in a grid so they do not all sit in one row
    cols = min(4, num_images)  # At most 4 images per row
    rows = math.ceil(num_images / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    
    # If there is only one row or column, keep axes iterable
    if rows > 1 or cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for i in range(num_images):
        ax = axes[i]
        
        # Convert the image for display
        img_np = images_tensor[i].cpu().permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        img_h, img_w = img_np.shape[0], img_np.shape[1]
        ax.imshow(img_np)
        
        # 1. Draw ground-truth boxes in green
        target_boxes = targets[i]["boxes"]
        if len(target_boxes) > 0:
            gt_boxes = xywh2xyxy(target_boxes.clone())
            gt_boxes[:, [0, 2]] *= img_w
            gt_boxes[:, [1, 3]] *= img_h
            for box in gt_boxes:
                x1, y1, x2, y2 = box.tolist()
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='green', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1-5, 'Actual', color='white', fontsize=10, backgroundcolor='green')

        # 2. Draw model predictions in red
        pred = preds[i]
        if len(pred) > 0:
            pred_boxes = pred[:, :4]
            pred_conf = pred[:, 4]
            for box, conf in zip(pred_boxes, pred_conf):
                x1, y1, x2, y2 = box.tolist()
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
                ax.add_patch(rect)
                ax.text(x1, y2+15, f'Pred: {conf:.2f}', color='white', fontsize=10, backgroundcolor='red')

        ax.axis('off')
        ax.set_title(f"Image Index: {start_idx + i}")

    # Hide unused grid cells
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

