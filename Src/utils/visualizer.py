import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from PIL import Image
import time
from tqdm.auto import tqdm
import numpy as np
from ultralytics.utils import nms



def plot_img(img, target=None, info=None, ax=None):
    """
    Display an object detection image with YOLO bounding boxes.

    Args:
        img: PIL Image or Tensor image.
        target: Dictionary containing:
                - boxes: bounding boxes in YOLO format
                         [x_center, y_center, width, height]
                         normalized between 0 and 1
                - labels: class IDs
        info: Optional information displayed below the image.
        ax: Optional matplotlib axes.
    """

    # Create figure if no axes were provided
    created_fig = False

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
        created_fig = True

    # Convert Tensor image to numpy if necessary
    if torch.is_tensor(img):
        img = img.permute(1, 2, 0).cpu().numpy()

    # Get image dimensions
    width, height = img.size if isinstance(img, Image.Image) else (img.shape[1], img.shape[0])

    # Display image
    ax.imshow(img)

    # Draw bounding boxes
    if target is not None:

        boxes = target.get("boxes", [])
        labels = target.get("labels", [])

        for box, class_id in zip(boxes, labels):

            # Convert tensor to list
            box = box.tolist()

            x_center, y_center, box_width, box_height = box

            # Convert YOLO normalized coordinates
            # to pixel coordinates
            x_center *= width
            y_center *= height
            box_width *= width
            box_height *= height

            # Convert center coordinates to top-left coordinates
            x_min = x_center - box_width / 2
            y_min = y_center - box_height / 2

            # Create rectangle
            rectangle = patches.Rectangle((x_min, y_min), box_width, box_height, linewidth=2, edgecolor="red", facecolor="none")

            ax.add_patch(rectangle)

            # Class name
            class_name = "truck" if int(class_id) == 0 else str(int(class_id))

            # Display class name
            ax.text(x_min, y_min - 5, class_name, fontsize=10, color="white", bbox=dict(facecolor="red", alpha=0.7))

    # Add optional information
    if info:
        ax.text(0.5, -0.05, info, transform=ax.transAxes, ha="center", va="top", fontsize=10)

    ax.axis("off")

    # Show figure only if we created it
    if created_fig:
        plt.show()



def measure_average_epoch_time(loader, device):
    """
    Measure average DataLoader epoch time.

    This benchmark measures data loading only.
    It does not perform model training.
    """

    epoch_times = []

    total_epochs = 5
    warmup_epochs = 2

    for epoch in tqdm(
        range(total_epochs),
        desc="Overall Progress"
    ):

        start_time = time.time()

        for images, targets in loader:
            # We only measure DataLoader performance.
            # No model computation is performed.
            pass

        end_time = time.time()

        current_time = end_time - start_time
        epoch_times.append(current_time)

    print("")

    for epoch, current_time in enumerate(epoch_times):

        warmup_note = "(warm-up)" if epoch < warmup_epochs else ""

        print(
            f"Epoch {epoch + 1}/{total_epochs} | "
            f"Time: {current_time:.2f} seconds {warmup_note}"
        )

    measured_epochs = epoch_times[warmup_epochs:]

    average_time = np.mean(measured_epochs)

    num_measured_epochs = total_epochs - warmup_epochs

    print(
        f"\nAverage execution time "
        f"(avg of last {num_measured_epochs}): "
        f"{average_time:.2f} seconds\n"
    )

    return average_time







