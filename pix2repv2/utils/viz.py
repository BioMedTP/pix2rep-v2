import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap


def pix2rep_prediction_figure(
    mri_slice: torch.Tensor, predicted_mask: torch.Tensor, gt_mask: torch.Tensor
) -> plt.Figure:
    """Create a visualization where the predicted segmentation are shown on one side of
    the image (overlaid in the original 2D image), and the GT segmentation on the other.

    Args:
        mri_slice (torch.Tensor): original MRI slice that has been segmented (shape [H, W])
        predicted_mask (torch.Tensor): predicted segmentation mask (shape [H, W])
        gt_mask (torch.Tensor): ground truth segmentation mask (shape [H, W])

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    mri_slice = mri_slice.cpu().numpy()
    predicted_mask = predicted_mask.cpu().numpy()
    gt_mask = gt_mask.cpu().numpy()

    class_labels = {0: "BACKGROUND", 1: "RV", 2: "MYO", 3: "LV"}
    colors = {0: "black", 1: "salmon", 2: "lime", 3: "royalblue"}

    # Create discrete colormap
    cmap = ListedColormap([colors[i] for i in range(4)])

    fig = plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title("Predicted segmentation")
    masked_mask = np.ma.masked_where(predicted_mask == 0, predicted_mask)
    plt.imshow(mri_slice.swapaxes(0, 1), cmap="gray")
    plt.imshow(
        masked_mask.swapaxes(0, 1),
        cmap=cmap,
        alpha=0.5,
        interpolation="none",
        vmin=0,
        vmax=3,
    )
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("GT segmentation")
    masked_mask = np.ma.masked_where(gt_mask == 0, gt_mask)
    plt.imshow(mri_slice.swapaxes(0, 1), cmap="gray")
    plt.imshow(
        masked_mask.swapaxes(0, 1),
        cmap=cmap,
        alpha=0.5,
        interpolation="none",
        vmin=0,
        vmax=3,
    )
    plt.axis("off")

    legend_elements = [
        Patch(facecolor=colors[i], label=class_labels[i])
        for i in range(1, 4)  # Skip background (0)
    ]
    fig.legend(handles=legend_elements, loc="center right", bbox_to_anchor=(1.0, 0.5))

    plt.tight_layout()

    return fig


def pix2rep_prediction_figure_for_miccai(
    mri_slice: torch.Tensor,
    predicted_mask: torch.Tensor,
    gt_mask: torch.Tensor,
) -> plt.Figure:
    """TODO"""

    mri_slice = mri_slice.detach().cpu().numpy()
    predicted_mask = predicted_mask.detach().cpu().numpy()
    gt_mask = gt_mask.detach().cpu().numpy()

    mri_slice = mri_slice.swapaxes(0, 1)
    predicted_mask = predicted_mask.swapaxes(0, 1)
    gt_mask = gt_mask.swapaxes(0, 1)

    p1, p99 = np.percentile(mri_slice, (1, 99))
    mri_slice = np.clip(mri_slice, p1, p99)
    mri_slice = (mri_slice - mri_slice.min()) / (
        mri_slice.max() - mri_slice.min() + 1e-8
    )

    colors = {
        0: (0, 0, 0, 0),  # transparent background
        1: (1.0, 0.4, 0.4, 1.0),  # RV
        2: (0.4, 1.0, 0.4, 1.0),  # MYO
        3: (0.3, 0.7, 1.0, 1.0),  # LV
    }

    cmap = ListedColormap([colors[i] for i in range(4)])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    axes[0].imshow(mri_slice, cmap="gray", interpolation="bicubic")

    masked_pred = np.ma.masked_where(predicted_mask == 0, predicted_mask)
    axes[0].imshow(
        masked_pred,
        cmap=cmap,
        alpha=0.45,  # softer overlay (more aesthetic)
        interpolation="bilinear",
        vmin=0,
        vmax=3,
    )

    axes[0].contour(predicted_mask == 1, colors=[colors[1]], linewidths=1.2)  # RV
    axes[0].contour(predicted_mask == 2, colors=[colors[2]], linewidths=1.2)  # MYO
    axes[0].contour(predicted_mask == 3, colors=[colors[3]], linewidths=1.4)  # LV

    axes[0].set_title("Prediction", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(mri_slice, cmap="gray", interpolation="bicubic")

    masked_gt = np.ma.masked_where(gt_mask == 0, gt_mask)
    axes[1].imshow(
        masked_gt,
        cmap=cmap,
        alpha=0.45,
        interpolation="bilinear",
        vmin=0,
        vmax=3,
    )

    axes[1].contour(gt_mask == 1, colors=[colors[1]], linewidths=1.2)
    axes[1].contour(gt_mask == 2, colors=[colors[2]], linewidths=1.2)
    axes[1].contour(gt_mask == 3, colors=[colors[3]], linewidths=1.4)

    axes[1].set_title("Ground Truth", fontsize=13, fontweight="bold")
    axes[1].axis("off")

    legend_elements = [
        Patch(facecolor=colors[1], edgecolor="none", label="RV"),
        Patch(facecolor=colors[2], edgecolor="none", label="MYO"),
        Patch(facecolor=colors[3], edgecolor="none", label="LV"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, -0.02),
    )

    return fig


def pix2rep_prediction_figure_3d(
    mri_slice: torch.Tensor, predicted_mask: torch.Tensor, gt_mask: torch.Tensor
):
    mri_slice = mri_slice.cpu().numpy()
    predicted_mask = predicted_mask.cpu().numpy()
    gt_mask = gt_mask.cpu().numpy()

    class_labels = {
        0: "background",
        1: "spleen",
        2: "right kidney",
        3: "left kidney",
        4: "gall bladder",
        5: "esophagus",
        6: "liver",
        7: "stomach",
        8: "aorta",
        9: "postcava",
        10: "pancreas",
        11: "right adrenal gland",
        12: "left adrenal gland",
        13: "duodenum",
        14: "bladder",
        15: "prostate/uterus",
    }
    colors = {
        0: "#000000",  # background — black
        1: "#ff6f61",  # spleen — coral
        2: "#32cd32",  # right kidney — lime green
        3: "#1e90ff",  # left kidney — dodger blue
        4: "#ffd700",  # gall bladder — gold
        5: "#8a2be2",  # esophagus — blue violet
        6: "#ff8c00",  # liver — dark orange
        7: "#00ced1",  # stomach — dark turquoise
        8: "#dc143c",  # aorta — crimson
        9: "#ff1493",  # postcava — deep pink
        10: "#20b2aa",  # pancreas — light sea green
        11: "#a0522d",  # right adrenal — sienna
        12: "#cd853f",  # left adrenal — peru
        13: "#7fff00",  # duodenum — chartreuse
        14: "#00bfff",  # bladder — deep sky blue
        15: "#ff00ff",  # prostate/uterus — magenta
    }

    # Create discrete colormap
    cmap = ListedColormap([colors[i] for i in range(len(class_labels))])

    fig = plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title("Predicted segmentation")
    masked_mask = np.ma.masked_where(predicted_mask == 0, predicted_mask)
    plt.imshow(mri_slice, cmap="gray")
    plt.imshow(
        masked_mask,
        cmap=cmap,
        alpha=0.5,
        interpolation="none",
        vmin=0,
        vmax=len(class_labels) - 1,
    )
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("GT segmentation")
    masked_mask = np.ma.masked_where(gt_mask == 0, gt_mask)
    plt.imshow(mri_slice, cmap="gray")
    plt.imshow(
        masked_mask,
        cmap=cmap,
        alpha=0.5,
        interpolation="none",
        vmin=0,
        vmax=len(class_labels) - 1,
    )
    plt.axis("off")

    legend_elements = [
        Patch(facecolor=colors[i], label=class_labels[i])
        for i in range(1, len(class_labels))  # Skip background (0)
    ]
    # fig.legend(handles=legend_elements, loc="center right", bbox_to_anchor=(1.0, 0.5))
    fig.legend(
        handles=legend_elements,
        loc="center right",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=8,  # smaller text
        ncol=1,  # split into columns (try 2–3)
        handlelength=0.8,  # smaller color boxes
        handleheight=0.8,
        labelspacing=0.3,  # tighter vertical spacing
        borderpad=0.3,
        columnspacing=0.8,
    )

    plt.tight_layout()

    return fig
