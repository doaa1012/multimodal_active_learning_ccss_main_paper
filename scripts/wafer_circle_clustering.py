import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle

def display_image(image, title="Image", cmap=None):
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()


def adjust_saturation_contrast(image, saturation_factor, contrast_factor):
    """Enhance saturation and contrast."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= saturation_factor
    hsv[..., 2] *= contrast_factor
    hsv[..., 1:] = np.clip(hsv[..., 1:], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def calculate_centroid(mask):
    """Calculate the centroid of a binary mask."""
    moments = cv2.moments(mask)
    if moments['m00'] == 0:
        return None
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])
    return (cx, cy)



def process_wafer_image_saturation_black(image_path, save_folder="output_clusters", k=18, top_n=5):
    """Clean wafer image, enhance, apply clustering, save centroids individually and collectively."""

    os.makedirs(save_folder, exist_ok=True)

    # Load and validate image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)

    # GrabCut for object segmentation
    rect = (10, 10, img.shape[1] - 20, img.shape[0] - 20)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest_contour = max(contours, key=cv2.contourArea)
    final_mask = np.zeros_like(mask2)
    cv2.drawContours(final_mask, [biggest_contour], -1, 1, thickness=cv2.FILLED)

    # Plot and save the mask to verify
    plt.figure(figsize=(6,6))
    plt.imshow(final_mask, cmap='gray')
    plt.title("Final Wafer Mask (After GrabCut + Contour)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "wafer_final_mask.png"), dpi=300)
    plt.close()

    ys, xs = np.where(final_mask == 1)
    top, bottom = np.min(ys), np.max(ys)
    left, right = np.min(xs), np.max(xs)

    cleaned_img = img.copy()
    cleaned_img[final_mask == 0] = (255, 255, 255)
    cropped_img = cleaned_img[top:bottom, left:right]

    # Enhanced versions
    img_versions = [
        ("Saturation and Contrast +", adjust_saturation_contrast(cropped_img, 1.25, 1.3)),
        ("Saturation and Contrast ++", adjust_saturation_contrast(cropped_img, 1.5, 1.6)),
        ("Saturation and Contrast +++", adjust_saturation_contrast(cropped_img, 1.75, 1.9))
    ]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    all_centroids_dict = {}

    for title, image in img_versions:
        pixel_values = image.reshape((-1, 3)).astype(np.float32)
        non_background_pixels = np.any(pixel_values < 245, axis=1)
        filtered_pixel_values = pixel_values[non_background_pixels]
        cv2.setRNGSeed(42)
        _, labels, centers = cv2.kmeans(filtered_pixel_values, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

        centers = np.uint8(centers)

        labels_full = np.full((pixel_values.shape[0]), -1)
        labels_full[non_background_pixels] = labels.flatten()
        labels_full = labels_full.reshape(image.shape[:2])

        unique, counts = np.unique(labels, return_counts=True)
        cluster_pixel_counts = dict(zip(unique, counts))
        sorted_clusters = sorted(cluster_pixel_counts, key=cluster_pixel_counts.get, reverse=True)[:top_n]

        fig, axs = plt.subplots(1, top_n, figsize=(5 * top_n, 5))
        centroids = []

        for idx, cluster in enumerate(sorted_clusters):
            cluster_mask = (labels_full == cluster).astype(np.uint8)
            cluster_image = np.zeros_like(image)
            cluster_image[cluster_mask == 1] = centers[cluster]

            centroid = calculate_centroid(cluster_mask)
            if centroid:
                # Correct centroid to full image coordinates
                corrected_centroid = (centroid[0] + left, centroid[1] + top)
                centroids.append(corrected_centroid)
                axs[idx].scatter(centroid[0], centroid[1], color='red', s=50)

            axs[idx].imshow(cv2.cvtColor(cluster_image, cv2.COLOR_BGR2RGB))
            axs[idx].axis('off')
            axs[idx].set_title(f'{title} - Cluster {idx + 1}')

            # Save each cluster image
            safe_title = title.replace(' ', '_')
            cluster_filename = f"{safe_title}_cluster_{idx+1}.png"
            cv2.imwrite(os.path.join(save_folder, cluster_filename), cluster_image)

        # Save top-N clusters combined
        plt.tight_layout()
        fig_filename = f"{safe_title}_top{top_n}_clusters.png"
        fig.savefig(os.path.join(save_folder, fig_filename))
        plt.show()

        # Save individual centroids
        centroid_df = pd.DataFrame(centroids, columns=['cx', 'cy'])
        centroid_csv = os.path.join(save_folder, f"{safe_title}_centroids.csv")
        centroid_pkl = os.path.join(save_folder, f"{safe_title}_centroids.pkl")
        centroid_df.to_csv(centroid_csv, index=False)
        with open(centroid_pkl, 'wb') as f:
            pickle.dump(centroids, f)

        # Add to dictionary for global save
        all_centroids_dict[safe_title] = centroids

    # Save merged centroids dictionary (Pickle)
    all_centroids_pkl = os.path.join(save_folder, "all_centroids_for_mapping.pkl")
    with open(all_centroids_pkl, "wb") as f:
        pickle.dump(all_centroids_dict, f)

    # Save merged centroids as flattened CSV
    all_centroids_flat = []
    for label, points in all_centroids_dict.items():
        for i, (cx, cy) in enumerate(points):
            all_centroids_flat.append({
                "version": label,
                "cluster": i + 1,
                "cx": cx,
                "cy": cy
            })
    all_centroids_csv = os.path.join(save_folder, "all_centroids_for_mapping.csv")
    pd.DataFrame(all_centroids_flat).to_csv(all_centroids_csv, index=False)

     