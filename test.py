import argparse
import os
import cv2
import yaml
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import datasets
import csv
import models
import utils
from utils import *
from torchvision import transforms
from mmcv.runner import load_checkpoint
import csv
import statistics as st
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
# from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import random
from models.sam2.build_sam import build_sam2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def get_annotations(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask

    return img


def calculate_metrics(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()

    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    tn = np.sum((pred == 0) & (gt == 0))

    epsilon = 1e-7  # To avoid division by zero

    dice = (2 * tp + epsilon) / (2 * tp + fp + fn + epsilon)
    iou = (tp + epsilon) / (tp + fp + fn + epsilon)
    precision = (tp + epsilon) / (tp + fp + epsilon)
    recall = (tp + epsilon) / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    return dice, iou, precision, recall, f1


def save_metrics_to_file(all_metrics, average_metrics, iou_list, dice_list, precision_list, recall_list, f1_list,
                         filename='all_metrics.txt'):
    """
    Save individual image metrics and also lists of metrics at the end.

    all_metrics: List of dictionaries containing metrics per image
    average_metrics: Dictionary containing average metrics
    iou_list, dice_list, precision_list, recall_list, f1_list: Lists to store metrics for all images
    filename: Path of the file to save the metrics
    """
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'dice', 'iou', 'precision', 'recall', 'f1'])
        writer.writeheader()

        # Save individual image metrics
        for metric in all_metrics:
            writer.writerow(metric)

        # Add a separator
        writer.writerow({k: '-' * 10 for k in writer.fieldnames})

        # Add average metrics
        writer.writerow({
            'filename': 'AVERAGE',
            'dice': f"{average_metrics['dice']:.4f}",
            'iou': f"{average_metrics['iou']:.4f}",
            'precision': f"{average_metrics['precision']:.4f}",
            'recall': f"{average_metrics['recall']:.4f}",
            'f1': f"{average_metrics['f1']:.4f}"
        })

        # Add lists of all metrics at the end
        f.write("\n\nIoU List:\n")
        f.write(", ".join([f"{iou:.4f}" for iou in iou_list]) + "\n")

        f.write("Dice List:\n")
        f.write(", ".join([f"{dice:.4f}" for dice in dice_list]) + "\n")

        f.write("Precision List:\n")
        f.write(", ".join([f"{precision:.4f}" for precision in precision_list]) + "\n")

        f.write("Recall List:\n")
        f.write(", ".join([f"{recall:.4f}" for recall in recall_list]) + "\n")

        f.write("F1 List:\n")
        f.write(", ".join([f"{f1:.4f}" for f1 in f1_list]) + "\n")


def eval_psnr(args, loader, model, star_writer, save_dir, verbose=False, org_image_size=(4096, 4096)):
    model.to(device)
    model.eval()

    metric_averagers = {
        'dice': utils.Averager(),
        'iou': utils.Averager(),
        'precision': utils.Averager(),
        'recall': utils.Averager(),
        'f1': utils.Averager()
    }

    pbar = tqdm(loader, leave=False, desc='val')

    all_metrics = []

    for idx, batch in enumerate(pbar):
        for k, v in batch.items():
            if k != 'filename':
                batch[k] = v.to(device)

        inp = batch['inp']
        gt = batch['gt']
        filenames = batch['filename']  # Get filenames from the batch

        pred = torch.sigmoid(model.infer(inp))
        pred = (pred > 0.5).float()

        for i in range(inp.shape[0]):
            # Convert to numpy arrays
            pred_np = pred[i].cpu().numpy()
            gt_np = gt[i].cpu().numpy()

            dice, iou, precision, recall, f1 = calculate_metrics(pred_np, gt_np)
            metrics = {
                'filename': filenames[i],
                'dice': dice,
                'iou': iou,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            all_metrics.append(metrics)

            for key, value in metrics.items():
                if key != 'filename':
                    metric_averagers[key].add(value, 1)

            generate_output(args, inp[i], gt[i], pred[i], filenames[i], star_writer, output_dir=save_dir,
                            org_image_size=org_image_size)

        if verbose:
            pbar.set_description('val Dice: {:.4f}, IoU: {:.4f}, F1: {:.4f}'.format(
                metric_averagers['dice'].item(),
                metric_averagers['iou'].item(),
                metric_averagers['f1'].item()
            ))

    average_metrics = {key: averager.item() for key, averager in metric_averagers.items()}
    return all_metrics, average_metrics


def count_circles_hough(mask, dp=1.2, min_dist=20, param1=50, param2=30, min_radius=5, max_radius=50):
    """
    Count circular particles in a mask using the Hough Circle Transform.

    Parameters:
    - mask: The mask image where circles are to be counted.
    - dp: Inverse ratio of the accumulator resolution to the image resolution (default: 1.2).
    - min_dist: Minimum distance between detected centers (default: 20).
    - param1: Higher threshold for the Canny edge detector (default: 50).
    - param2: Accumulator threshold for the circle centers at the detection stage (default: 30).
    - min_radius: Minimum circle radius (default: 5).
    - max_radius: Maximum circle radius (default: 50).

    Returns:
    - The number of detected circles.
    """
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
                               param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        return len(circles)
    return 0


def detect_circular_particles(mask_path, particle_radius=28):
    """
    Enhanced particle detection optimized for dense, connected regions.
    """
    # Read and preprocess mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (1024, 1024))
    if mask is None:
        raise ValueError(f"Could not read mask from {mask_path}")

    binary_mask = (mask > 127).astype(np.uint8)

    # Step 1: Enhanced Distance Transform
    distance = ndi.distance_transform_edt(binary_mask)

    # Step 2: Multi-scale Detection with Different Approaches
    all_coordinates = []

    # A) Multiple threshold levels with different sensitivities
    threshold_levels = [
        particle_radius * 0.05,  # Very sensitive
        particle_radius * 0.10,
        particle_radius * 0.15,
        particle_radius * 0.20,
        particle_radius * 0.25
    ]

    min_distances = [
        int(particle_radius * 0.3),  # More aggressive
        int(particle_radius * 0.4),
        int(particle_radius * 0.5)
    ]

    # Combine different thresholds and distances
    for threshold in threshold_levels:
        for min_dist in min_distances:
            coordinates = peak_local_max(
                distance,
                min_distance=min_dist,
                threshold_abs=threshold,
                exclude_border=False,
            )
            all_coordinates.extend(coordinates)

    # B) Use morphological operations to separate connected components
    #     kernel_size = int(particle_radius * 0.3)
    kernel_size = int(particle_radius * 0.8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded = cv2.erode(binary_mask, kernel, iterations=1)

    # Find additional centers from eroded image
    dist_eroded = ndi.distance_transform_edt(eroded)
    #     coordinates_eroded = peak_local_max(
    #         dist_eroded,
    #         min_distance=int(particle_radius * 0.4),
    #         threshold_abs=particle_radius * 0.1,
    #         exclude_border=False,
    #     )
    #     coordinates_eroded = peak_local_max(
    #         dist_eroded,
    #         min_distance=int(particle_radius * 0.05),
    #         threshold_abs=particle_radius * 0.05,
    #         exclude_border=False,
    #     )

    coordinates_eroded = peak_local_max(
        dist_eroded,
        min_distance=int(particle_radius * 0.1),
        threshold_abs=particle_radius * 0.1,
        exclude_border=False,
    )
    all_coordinates.extend(coordinates_eroded)

    # C) Use circular Hough transform for additional detection
    circles = cv2.HoughCircles(
        binary_mask,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=int(particle_radius * 0.8),
        param1=50,
        param2=15,
        minRadius=int(particle_radius * 0.6),
        maxRadius=int(particle_radius * 1.4)
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        for circle in circles:
            all_coordinates.append([circle[1], circle[0]])  # y, x coordinates

    # Remove duplicates with a distance threshold
    filtered_coordinates = []
    distance_threshold = particle_radius * 0.5

    all_coordinates = np.unique(all_coordinates, axis=0)
    for coord in all_coordinates:
        if not any(np.hypot(coord[0] - existing[0], coord[1] - existing[1]) < distance_threshold
                   for existing in filtered_coordinates):
            filtered_coordinates.append(coord)

    # Create markers for watershed
    markers = np.zeros_like(binary_mask, dtype=np.int32)
    for i, coord in enumerate(filtered_coordinates, start=1):
        markers[coord[0], coord[1]] = i

    # Apply watershed
    labels = watershed(-distance, markers, mask=binary_mask)

    # Process regions and validate particles
    valid_centers = []
    processed_regions = np.zeros_like(binary_mask)

    for label_idx in range(1, labels.max() + 1):
        particle_mask = (labels == label_idx).astype(np.uint8)
        contours, _ = cv2.findContours(
            particle_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        contour = contours[0]
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

        expected_area = np.pi * particle_radius * particle_radius
        #         min_area = expected_area * 0.2  # More permissive
        min_area = expected_area * 0.05  # More permissive
        max_area = expected_area * 2.5  # More permissive

        is_circular = circularity > 0.4  # More permissive
        is_valid_size = min_area <= area <= max_area

        if is_valid_size and (is_circular or area < expected_area * 1.3):
            M = cv2.moments(particle_mask)
            if M["m00"] != 0:
                cy = int(M["m10"] / M["m00"])
                cx = int(M["m01"] / M["m00"])
                valid_centers.append((cx, cy))
                processed_regions = cv2.bitwise_or(processed_regions, particle_mask)

    # Final pass for missed regions
    remaining_mask = cv2.bitwise_and(binary_mask, cv2.bitwise_not(processed_regions))
    if np.any(remaining_mask):
        contours, _ = cv2.findContours(
            remaining_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cy = int(M["m10"] / M["m00"])
                    cx = int(M["m01"] / M["m00"])
                    is_new_center = all(
                        np.hypot(cx - c[0], cy - c[1]) > particle_radius * 0.8
                        for c in valid_centers
                    )
                    if is_new_center:
                        valid_centers.append((cx, cy))

    return binary_mask, distance, labels, valid_centers


def generate_output(args, org_image, gt, pred_mask, filename, star_writer, output_dir, org_image_size=(4096, 4096)):
    height, width = org_image.shape[1], org_image.shape[2]

    original_image = org_image.detach().cpu().permute(1, 2, 0).numpy()
    original_image = cv2.resize(original_image,
                                (config['model']['args']['inp_size'], config['model']['args']['inp_size']))
    original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())

    original_mask = gt.detach().cpu().squeeze().numpy()
    original_mask_8_bit = (original_mask * 255).astype(np.uint8)
    # Count circles in the ground truth mask using Hough Circle Transform
    ground_truth_circular_count = count_circles_hough(original_mask_8_bit)

    predicted_mask = pred_mask.detach().cpu().numpy().reshape(config['model']['args']['inp_size'],
                                                              config['model']['args']['inp_size'])

    predicted_mask = np.clip(predicted_mask, 0, 1)

    # add this lines of code to get the coordinates for CryoSparc
    predicted_mask = np.rot90(predicted_mask, k=3)
    predicted_mask = predicted_mask.T
    predicted_mask_8bit = (predicted_mask * 255).astype(np.uint8)

    # Save temporary mask image
    temp_mask_path = os.path.join(output_dir, "temp_mask.png")
    cv2.imwrite(temp_mask_path, predicted_mask_8bit)

    binary_mask, distance, labels, valid_centers = detect_circular_particles(
        temp_mask_path,
        particle_radius=args.particle_radius
    )

    # Remove temporary file
    os.remove(temp_mask_path)

    filename_base = filename.rsplit('_', 1)[0]
    filename_mrc = filename_base + '.mrc'
    #    filename_mrc = filename + '.mrc'

    #    original_height, original_width = org_image_size[0], org_image_size[1]
    original_width, original_height = org_image_size[0], org_image_size[1]

    # Compute scaling factors for x and y axes
    scale_x = original_width / width
    scale_y = original_height / height

    original_image = np.rot90(original_image, k=3)
    original_image = original_image.T
    original_image = np.transpose(original_image, (1, 2, 0))

    original_mask = np.rot90(original_mask, k=3)
    original_mask = original_mask.T

    # Create visualization image

    mask_bgr = cv2.cvtColor(predicted_mask_8bit, cv2.COLOR_GRAY2BGR)

    # Draw circles and write to star file
    circle_count = 0
    for center in valid_centers:
        cx, cy = center

        # Draw circle on visualization
        cv2.circle(mask_bgr, (cy, cx), args.particle_radius, (0, 0, 255), 3)

        # Scale coordinates to original image size
        scaled_x = int(round(cy * scale_x))
        scaled_y = int(round(cx * scale_y))
        scaled_diameter = int(round(2 * args.particle_radius * (scale_x + scale_y) / 2))

        # Write to star file
        star_writer.writerow([
            filename_mrc,
            scaled_x,
            scaled_y,
            scaled_diameter
        ])
        circle_count += 1

    mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)

    # Plotting
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(original_mask, cmap='gray')
    axes[1].set_title(f"Original Mask:{ground_truth_circular_count}")
    axes[1].axis('off')

    axes[2].imshow(predicted_mask, cmap='gray')
    axes[2].set_title("Predicted Mask")
    axes[2].axis('off')

    axes[3].imshow(mask_rgb)
    axes[3].axis('off')
    axes[3].set_title(f"Predicted Mask with circles: {circle_count}")

    plt.tight_layout(pad=2.0)
    plt.savefig(f"{save_dir}/{filename}.png")
    plt.close()


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def override_config(config, args):
    """Override parts of config with user args (if provided)."""
    # Training dataset overrides
    if args.test_images is not None:
        config["test_dataset"]["dataset"]["args"]["root_path_1"] = args.test_images
    if args.test_labels is not None:
        config["test_dataset"]["dataset"]["args"]["root_path_2"] = args.test_labels

    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/cod-sam-vit-l.yaml")
    parser.add_argument('--model',
                        default="./checkpoints/5_shot/10028_few_shot/model_epoch_best.pth", help='path to  the trained model')
    parser.add_argument('--shot', type=int, default=5, choices=[1, 5, 10], required=False,
                        help='Number of shots for few-shot training (1, 5, or 10)')
    parser.add_argument('--protein_name', default='10028', help='EMPAIR ID of the protein')
    parser.add_argument('--particle_radius', type=int, default=28, help='radius of the particle for given protein having images of size (1024, 1024)')
    parser.add_argument('--org_image_size', type=int, nargs=2, default=[4096, 4096], help='size of original images')
    parser.add_argument('--output_dir', default='./data_outputs',
                        help='save the output images and information')

    # Arguments for test images and labels
    parser.add_argument('--test_images', type=str,
                        default="./data/test/images/",
                        help="Path to test/images")
    parser.add_argument('--test_labels', type=str,
                        default="./data/test/labels/",
                        help="Path to test/labels")

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = override_config(config, args)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # Access the image size as a tuple
    image_size = tuple(args.org_image_size)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        num_workers=1, shuffle=False)

    model = models.make(config['model'])
    sam_checkpoint = torch.load(args.model, map_location='cpu')

    # Extract model state dict from checkpoint
    if isinstance(sam_checkpoint, dict) and 'model' in sam_checkpoint:
        # If checkpoint is a dictionary containing 'model' key
        model_state_dict = sam_checkpoint['model']
    else:
        # If checkpoint is directly the state dict
        model_state_dict = sam_checkpoint

    # Load the model state dict
    model.load_state_dict(model_state_dict, strict=True)

    shot_dir = f'{args.shot}_shot'
    protein_path = os.path.join(args.output_dir, shot_dir, args.protein_name)
    os.makedirs(protein_path, exist_ok=True)

    save_dir = os.path.join(protein_path, 'micrographs_outputs')
    os.makedirs(save_dir, exist_ok=True)

    star_file = os.path.join(protein_path, f'{args.protein_name}_star_file.star')

    with open(star_file, 'w', newline='') as star_file:
        star_writer = csv.writer(star_file, delimiter=' ')
        star_writer.writerow([])
        star_writer.writerow(["data_"])
        star_writer.writerow([])
        star_writer.writerow(["loop_"])
        star_writer.writerow(["_rlnMicrographName", "#1"])
        star_writer.writerow(["_rlnCoordinateX", "#2"])
        star_writer.writerow(["_rlnCoordinateY", "#3"])
        star_writer.writerow(["_rlnDiameter", "#4"])

        all_metrics, average_metrics = eval_psnr(args, loader=loader, model=model, star_writer=star_writer,
                                                 verbose=True, save_dir=save_dir, org_image_size=image_size)

        # Lists to hold the metrics for all images
        iou_list = []
        dice_list = []
        precision_list = []
        recall_list = []
        f1_list = []

        # Populate the metric lists from the per-image metrics
        for metric in all_metrics:
            iou_list.append(metric['iou'])
            dice_list.append(metric['dice'])
            precision_list.append(metric['precision'])
            recall_list.append(metric['recall'])
            f1_list.append(metric['f1'])

        # Save individual image metrics and lists of metrics to the same file
        save_metrics_to_file(all_metrics, average_metrics, iou_list, dice_list, precision_list, recall_list,
                             f1_list, os.path.join(protein_path, 'all_metrics.txt'))

        # Print average metrics
        print('Average Test Metrics:')
        for key, value in average_metrics.items():
            print(f'{key.capitalize()}: {value:.4f}')