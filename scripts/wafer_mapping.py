import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def map_centroids_to_wafer(image_path, csv_path, centroids, 
                            x_center, y_center, wafer_radius_pixels,
                            save_path="mapped_centroids_plot.png", show_plot=True):
    """Map centroids onto the wafer using stage coordinate transformation."""

    # Load Image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create Circle Mask
    mask = np.zeros_like(gray)
    cv2.circle(mask, (x_center, y_center), wafer_radius_pixels, 255, thickness=-1)

    # Load CSV (Stage Coordinates)
    df = pd.read_csv(csv_path)
    x_stage = df['x'].values
    y_stage = df['y'].values

    # Normalize Stage Coordinates
    x_center_stage = (x_stage.max() + x_stage.min()) / 2
    y_center_stage = (y_stage.max() + y_stage.min()) / 2
    x_range = x_stage.max() - x_stage.min()
    y_range = y_stage.max() - y_stage.min()

    x_norm = (x_stage - x_center_stage) / (x_range / 2)
    y_norm = (y_stage - y_center_stage) / (y_range / 2)

    # Scale to pixel wafer space
    x_scaled = x_norm * wafer_radius_pixels
    y_scaled = y_norm * wafer_radius_pixels

    x_pixel = x_center + x_scaled
    y_pixel = y_center - y_scaled  # Flip Y-axis

    # Filter points inside wafer
    distances = np.sqrt((x_pixel - x_center) ** 2 + (y_pixel - y_center) ** 2)
    inside_circle = distances <= wafer_radius_pixels

    x_filtered = x_pixel[inside_circle]
    y_filtered = y_pixel[inside_circle]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_rgb)

    # Stage points
    ax.scatter(x_filtered, y_filtered, c='red', s=10, label='Stage Points')

    # Centroids
    for idx, (cx, cy) in enumerate(centroids):
        ax.scatter(cx, cy, color='blue', s=100, marker='x', zorder=3)
        ax.text(cx + 10, cy - 10, f'C{idx+1}', fontsize=10, color='white', weight='bold', zorder=4)

    # Wafer circle
    wafer_circle = plt.Circle((x_center, y_center), wafer_radius_pixels, color='yellow', fill=False, linewidth=2)
    ax.add_patch(wafer_circle)

    ax.legend()
    ax.set_title("Mapped Stage Points + Cluster Centroids on Wafer")
    ax.axis('off')

    # Adjust layout properly before saving
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=300)

    if show_plot:
        plt.show()
    plt.close(fig)

    
