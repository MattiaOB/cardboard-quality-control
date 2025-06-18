"""
Data Augmentation + Synthetic Defect Generation System
File: algorithms/augmentation.py

FEATURES:
- Augmenting good images
- Augmenting defective images
- Synthetic defects confined exclusively to cardboard area
"""

import os
import cv2
import numpy as np
import random
from PIL import Image
import albumentations as A
import shutil
from typing import List, Tuple
from tqdm import tqdm

# Import unified contour detection module
try:
    from .contour_detection import CardboardContourDetector
except ImportError:
    from contour_detection import CardboardContourDetector


class CompleteAugmentationSystem:
    """
    Complete augmentation system.
    Generates synthetic defects exclusively within detected cardboard area.
    """

    def __init__(self, input_folder=None, output_folder="dataset_crop"):
        # Setup paths
        if input_folder is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            input_folder = os.path.join(project_root, "dataset_crop")

        self.input_folder = os.path.abspath(input_folder)
        self.output_folder = os.path.abspath(output_folder)
        self.good_folder = os.path.join(self.input_folder, "good")
        self.defective_folder = os.path.join(self.input_folder, "defective")

        # Output folders
        self.output_good = os.path.join(output_folder, "good")
        self.output_defective = os.path.join(output_folder, "defective")
        self.output_synthetic = os.path.join(output_folder, "synthetic_defective")

        # Initialize unified contour detector
        self.contour_detector = CardboardContourDetector(debug_mode=False)

        # Setup augmentation strategies
        self.setup_augmentation_strategies()

    def setup_augmentation_strategies(self):
        """Setup augmentation pipelines"""

        # Lighting variations
        self.lighting_augmentation = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.8),
            A.RandomGamma(gamma_limit=(80, 120), p=0.7),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
        ])

        # Geometric transformations
        self.geometric_augmentation = A.Compose([
            A.Affine(
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                scale=(0.9, 1.1), rotate=(-15, 15), p=1.0
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
        ])

        # Optical distortions
        self.optical_augmentation = A.Compose([
            A.OpticalDistortion(distort_limit=0.2, p=0.8),
            A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.6),
            A.ElasticTransform(alpha=1, sigma=50, p=0.4),
        ])

        # Noise and blur
        try:
            noise_transform = A.GaussNoise(noise_scale_factor=0.1, p=0.6)
        except Exception:
            noise_transform = A.GaussNoise(var_limit=(10.0, 50.0), p=0.6)

        self.noise_augmentation = A.Compose([
            noise_transform,
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.4),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.3),
            A.Blur(blur_limit=3, p=0.4),
            A.MotionBlur(blur_limit=3, p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.3),
        ])

    def create_output_structure(self):
        """Create output folder structure"""
        folders = [self.output_folder, self.output_good, self.output_defective, self.output_synthetic]

        for folder in folders:
            if os.path.exists(folder):
                shutil.rmtree(folder)
            os.makedirs(folder, exist_ok=True)

        return True

    def load_images_from_folder(self, folder_path: str) -> List[Tuple[np.ndarray, str]]:
        """Load images from folder"""
        images = []
        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

        if not os.path.exists(folder_path):
            return images

        for filename in os.listdir(folder_path):
            if filename.lower().endswith(extensions):
                filepath = os.path.join(folder_path, filename)
                try:
                    img = cv2.imread(filepath)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        images.append((img, filename))
                except Exception:
                    continue

        return images

    def apply_traditional_augmentation(self, images: List[Tuple[np.ndarray, str]],
                                     category: str, multiplier: int = 20) -> int:
        """Apply traditional augmentation techniques"""
        output_folder = self.output_good if category == 'good' else self.output_defective
        strategies = [
            ('lighting', self.lighting_augmentation),
            ('geometric', self.geometric_augmentation),
            ('optical', self.optical_augmentation),
            ('noise', self.noise_augmentation),
        ]

        generated_count = 0

        for img, original_filename in tqdm(images, desc=f"Augmenting {category}"):
            base_name = os.path.splitext(original_filename)[0]

            # Save original copy
            original_path = os.path.join(output_folder, f"{base_name}_original.jpg")
            cv2.imwrite(original_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            generated_count += 1

            # Generate variants
            variants_per_strategy = max(1, multiplier // len(strategies))

            for strategy_name, strategy_pipeline in strategies:
                for variant_idx in range(variants_per_strategy):
                    try:
                        if img is None or img.size == 0:
                            continue

                        augmented = strategy_pipeline(image=img)['image']

                        if augmented is None or augmented.size == 0:
                            continue

                        aug_filename = f"{base_name}_{strategy_name}_{variant_idx:03d}.jpg"
                        aug_path = os.path.join(output_folder, aug_filename)

                        cv2.imwrite(aug_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
                        generated_count += 1

                    except Exception:
                        continue

        return generated_count

    def generate_synthetic_defects(self, good_images: List[Tuple[np.ndarray, str]],
                                 num_synthetic: int = 150) -> int:
        """Generate synthetic defects confined to cardboard area"""
        if not good_images:
            return 0

        defect_types = ['scratch', 'tear', 'spot']
        per_type = num_synthetic // len(defect_types)
        remainder = num_synthetic % len(defect_types)

        generated_count = 0

        for type_idx, defect_type in enumerate(defect_types):
            current_count = per_type + (1 if type_idx < remainder else 0)

            for variant_idx in range(current_count):
                try:
                    img, original_filename = random.choice(good_images)
                    base_name = os.path.splitext(original_filename)[0]

                    defective_img = self._inject_synthetic_defect(img, defect_type)

                    if defective_img is None:
                        continue

                    synth_filename = f"synth_{defect_type}_{variant_idx:03d}_{base_name[:10]}.jpg"
                    synth_path = os.path.join(self.output_synthetic, synth_filename)

                    cv2.imwrite(synth_path, cv2.cvtColor(defective_img, cv2.COLOR_RGB2BGR))
                    generated_count += 1

                except Exception:
                    continue

        return generated_count

    def _inject_synthetic_defect(self, img: np.ndarray, defect_type: str) -> np.ndarray:
        """Inject synthetic defect using unified contour detection"""
        try:
            result_img = img.copy()
            h, w = img.shape[:2]

            if h == 0 or w == 0:
                return None

            # Detect cardboard contour using unified detector
            cardboard_contour = self.contour_detector.detect_cardboard_contour(img)

            if cardboard_contour is not None:
                # Create mask using unified module
                cardboard_mask = self.contour_detector.create_cardboard_mask(
                    img, cardboard_contour, padding_percent=5
                )

                if cardboard_mask is not None and np.sum(cardboard_mask > 0) > 100:
                    pass  # Use detected mask
                else:
                    cardboard_mask = self.contour_detector.create_safe_fallback_mask(img, strategy='auto')
            else:
                cardboard_mask = self.contour_detector.create_safe_fallback_mask(img, strategy='auto')

            # Ensure mask is usable
            valid_pixels = np.sum(cardboard_mask > 0)
            if valid_pixels < 50:
                cardboard_mask = self.contour_detector.create_safe_fallback_mask(img, strategy='center')

            # Generate defect within cardboard area only
            if defect_type == 'scratch':
                result_img = self._add_realistic_scratch_in_cardboard(result_img, cardboard_mask)
            elif defect_type == 'tear':
                result_img = self._add_realistic_tear_in_cardboard(result_img, cardboard_mask)
            elif defect_type == 'spot':
                result_img = self._add_realistic_spots_in_cardboard(result_img, cardboard_mask)
            else:
                return img

            return result_img

        except Exception:
            return None

    def _get_random_point_in_mask(self, mask: np.ndarray) -> tuple:
        """Get random point within mask"""
        valid_points = np.where(mask == 255)

        if len(valid_points[0]) == 0:
            h, w = mask.shape
            return (w // 2, h // 2)

        idx = random.randint(0, len(valid_points[0]) - 1)
        y = valid_points[0][idx]
        x = valid_points[1][idx]

        return (x, y)

    def _is_point_in_mask(self, x: int, y: int, mask: np.ndarray) -> bool:
        """Check if point is within mask"""
        h, w = mask.shape
        if 0 <= x < w and 0 <= y < h:
            return mask[y, x] == 255
        return False

    def _add_realistic_scratch_in_cardboard(self, img: np.ndarray, cardboard_mask: np.ndarray) -> np.ndarray:
        """Add realistic scratch confined to cardboard area"""
        h, w = img.shape[:2]
        result = img.copy().astype(np.float64)

        try:
            # Start point within cardboard
            start_x, start_y = self._get_random_point_in_mask(cardboard_mask)

            # Scratch parameters
            length = random.randint(40, min(w, h) // 3)
            angle = random.uniform(0, 2 * np.pi)
            base_thickness = random.randint(1, 3)

            num_segments = max(10, length // 5)
            segment_length = length / num_segments

            current_x, current_y = start_x, start_y

            for i in range(num_segments):
                # Next point with variations
                next_x = current_x + segment_length * np.cos(angle) + random.uniform(-2, 2)
                next_y = current_y + segment_length * np.sin(angle) + random.uniform(-2, 2)

                next_x = int(next_x)
                next_y = int(next_y)

                # Stop if leaving cardboard area
                if not self._is_point_in_mask(next_x, next_y, cardboard_mask):
                    break

                # Segment variations
                intensity_factor = random.uniform(0.2, 0.8)
                current_thickness = max(1, int(base_thickness * random.uniform(0.5, 1.5)))

                # Color based on local area
                if len(img.shape) == 3:
                    local_color = img[int(current_y), int(current_x)]
                    color_variation = random.uniform(0.3, 0.7)
                    scratch_color = local_color * color_variation

                    color_shift = np.array([
                        random.uniform(-20, 20),
                        random.uniform(-20, 20),
                        random.uniform(-20, 20)
                    ])
                    scratch_color = np.clip(scratch_color + color_shift, 0, 255)
                else:
                    local_intensity = img[int(current_y), int(current_x)]
                    scratch_color = local_intensity * random.uniform(0.3, 0.7)

                transparency = random.uniform(0.4, 0.9)

                # Draw segment
                segment_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.line(segment_mask, (int(current_x), int(current_y)),
                        (next_x, next_y), 255, current_thickness)

                # Apply only within cardboard area
                segment_mask = cv2.bitwise_and(segment_mask, cardboard_mask)

                mask_norm = segment_mask.astype(np.float64) / 255.0

                if len(img.shape) == 3:
                    mask_3d = np.stack([mask_norm, mask_norm, mask_norm], axis=-1)
                    scratch_color_3d = scratch_color.reshape(1, 1, -1)
                    result = result * (1 - mask_3d * transparency) + scratch_color_3d * mask_3d * transparency
                else:
                    result = result * (1 - mask_norm * transparency) + scratch_color * mask_norm * transparency

                current_x, current_y = next_x, next_y
                angle += random.uniform(-0.1, 0.1)

        except Exception:
            pass

        return np.clip(result, 0, 255).astype(np.uint8)

    def _add_realistic_tear_in_cardboard(self, img: np.ndarray, cardboard_mask: np.ndarray) -> np.ndarray:
        """Add realistic tear confined to cardboard area"""
        h, w = img.shape[:2]
        result = img.copy().astype(np.float64)

        try:
            # Start point within cardboard
            start_x, start_y = self._get_random_point_in_mask(cardboard_mask)

            # Tear parameters
            base_length = random.randint(30, min(w, h) // 4)
            angle = random.uniform(-np.pi / 4, np.pi / 4)
            base_thickness = random.randint(2, 6)

            num_segments = max(8, base_length // 4)
            segment_length = base_length / num_segments

            current_x, current_y = start_x, start_y

            for i in range(num_segments):
                # Next point with deviations
                deviation = random.uniform(-0.3, 0.3)
                current_angle = angle + deviation

                next_x = current_x + segment_length * np.cos(current_angle)
                next_y = current_y + segment_length * np.sin(current_angle) + random.uniform(-3, 3)

                next_x = int(next_x)
                next_y = int(next_y)

                # Stop if leaving cardboard area
                if not self._is_point_in_mask(next_x, next_y, cardboard_mask):
                    break

                # Segment variations
                depth_factor = random.uniform(0.1, 0.5)
                current_thickness = max(1, int(base_thickness * random.uniform(0.7, 1.3)))

                # Color based on depth
                if len(img.shape) == 3:
                    local_color = img[int(current_y), int(current_x)]
                    tear_color = local_color * depth_factor

                    color_noise = np.array([
                        random.uniform(-10, 10),
                        random.uniform(-10, 10),
                        random.uniform(-10, 10)
                    ])
                    tear_color = np.clip(tear_color + color_noise, 0, 255)
                else:
                    local_intensity = img[int(current_y), int(current_x)]
                    tear_color = local_intensity * depth_factor

                transparency = random.uniform(0.6, 0.95)

                # Draw segment
                segment_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.line(segment_mask, (int(current_x), int(current_y)),
                        (next_x, next_y), 255, current_thickness)

                # Apply only within cardboard area
                segment_mask = cv2.bitwise_and(segment_mask, cardboard_mask)

                mask_norm = segment_mask.astype(np.float64) / 255.0

                if len(img.shape) == 3:
                    mask_3d = np.stack([mask_norm, mask_norm, mask_norm], axis=-1)
                    tear_color_3d = tear_color.reshape(1, 1, -1)
                    result = result * (1 - mask_3d * transparency) + tear_color_3d * mask_3d * transparency
                else:
                    result = result * (1 - mask_norm * transparency) + tear_color * mask_norm * transparency

                current_x, current_y = next_x, next_y
                angle += random.uniform(-0.15, 0.15)

        except Exception:
            pass

        return np.clip(result, 0, 255).astype(np.uint8)

    def _add_realistic_spots_in_cardboard(self, img: np.ndarray, cardboard_mask: np.ndarray) -> np.ndarray:
        """Add realistic monochromatic spots confined to cardboard area"""
        h, w = img.shape[:2]
        result = img.copy().astype(np.float64)

        try:
            num_spots = random.randint(2, 6)

            for spot_idx in range(num_spots):
                # Position within cardboard
                spot_x, spot_y = self._get_random_point_in_mask(cardboard_mask)

                # Variable parameters
                base_radius = random.randint(3, 12)
                axis_a = random.randint(base_radius // 2, base_radius * 2)
                axis_b = random.randint(base_radius // 2, base_radius * 2)
                rotation_angle = random.randint(0, 180)

                # Calculate monochromatic color
                if len(img.shape) == 3:
                    local_region = img[max(0, spot_y - 5):min(h, spot_y + 5),
                                     max(0, spot_x - 5):min(w, spot_x + 5)]

                    if local_region.size > 0:
                        # Convert to grayscale for intensity calculation
                        local_gray = cv2.cvtColor(local_region.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                        local_intensity = np.mean(local_gray)
                    else:
                        local_intensity = 128

                    # Monochromatic spot types
                    spot_type = random.choice(['dark_spot', 'light_spot', 'medium_gray', 'black_spot', 'white_spot'])

                    if spot_type == 'dark_spot':
                        target_intensity = local_intensity * random.uniform(0.2, 0.5)
                    elif spot_type == 'light_spot':
                        target_intensity = min(255, local_intensity + random.uniform(30, 80))
                    elif spot_type == 'medium_gray':
                        target_intensity = random.uniform(80, 180)
                    elif spot_type == 'black_spot':
                        target_intensity = random.uniform(0, 40)
                    else:  # white_spot
                        target_intensity = random.uniform(200, 255)

                    intensity_noise = random.uniform(-15, 15)
                    target_intensity = np.clip(target_intensity + intensity_noise, 0, 255)

                    # Create gray color (same value for R, G, B)
                    spot_color = np.array([target_intensity, target_intensity, target_intensity])

                else:
                    # Already grayscale
                    local_intensity = np.mean(img[max(0, spot_y - 5):min(h, spot_y + 5),
                                              max(0, spot_x - 5):min(w, spot_x + 5)])

                    spot_type = random.choice(['darker', 'lighter', 'black', 'white', 'gray'])

                    if spot_type == 'darker':
                        spot_color = local_intensity * random.uniform(0.3, 0.7)
                    elif spot_type == 'lighter':
                        spot_color = min(255, local_intensity * random.uniform(1.2, 1.8))
                    elif spot_type == 'black':
                        spot_color = random.uniform(0, 40)
                    elif spot_type == 'white':
                        spot_color = random.uniform(200, 255)
                    else:  # gray
                        spot_color = random.uniform(80, 180)

                    spot_color = np.clip(spot_color, 0, 255)

                max_transparency = random.uniform(0.3, 0.8)

                # Elliptical mask
                spot_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.ellipse(spot_mask, (spot_x, spot_y), (axis_a, axis_b),
                           rotation_angle, 0, 360, 255, -1)

                # Apply only within cardboard area
                spot_in_cardboard = cv2.bitwise_and(spot_mask, cardboard_mask)

                if np.sum(spot_in_cardboard) < 10:
                    continue

                # Radial gradient for transparency
                Y, X = np.ogrid[:h, :w]
                dist_from_center = np.sqrt((X - spot_x) ** 2 + (Y - spot_y) ** 2)
                max_dist = max(axis_a, axis_b)

                normalized_dist = np.clip(dist_from_center / max_dist, 0, 1)
                transparency_gradient = max_transparency * (1 - normalized_dist ** 2)

                # Apply only in valid cardboard areas
                valid_area = (spot_in_cardboard > 0)
                transparency_map = np.zeros((h, w))
                transparency_map[valid_area] = transparency_gradient[valid_area]

                # Random variations for natural look
                noise_factor = random.uniform(0.8, 1.2)
                transparency_map *= noise_factor
                transparency_map = np.clip(transparency_map, 0, 1)

                # Apply monochromatic spot
                if len(img.shape) == 3:
                    transparency_3d = np.stack([transparency_map, transparency_map, transparency_map], axis=-1)
                    spot_color_3d = spot_color.reshape(1, 1, -1)
                    result = result * (1 - transparency_3d) + spot_color_3d * transparency_3d
                else:
                    result = result * (1 - transparency_map) + spot_color * transparency_map

        except Exception:
            pass

        return np.clip(result, 0, 255).astype(np.uint8)

    def run_complete_augmentation(self, good_multiplier: int = 25,
                                defective_multiplier: int = 15,
                                synthetic_count: int = 150):
        """Execute complete augmentation pipeline"""
        if not self.create_output_structure():
            return False

        # Load original images
        good_images = self.load_images_from_folder(self.good_folder)
        defective_images = self.load_images_from_folder(self.defective_folder)

        if not good_images and not defective_images:
            return False

        # PHASE 1: Traditional augmentation for good images
        if good_images:
            print(f"🟢 PHASE 1: Augmenting good images (x{good_multiplier})")
            self.apply_traditional_augmentation(good_images, 'good', good_multiplier)

        # PHASE 2: Traditional augmentation for defective images
        if defective_images:
            print(f"🔴 PHASE 2: Augmenting defective images (x{defective_multiplier})")
            self.apply_traditional_augmentation(defective_images, 'defective', defective_multiplier)

        # PHASE 3: Synthetic defect generation
        if good_images and synthetic_count > 0:
            print(f"🎨 PHASE 3: Generating {synthetic_count} synthetic defects")
            self.generate_synthetic_defects(good_images, synthetic_count)

        return True


def main_augmentation_pipeline():
    """Main augmentation pipeline execution"""
    # Configuration
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir.endswith('algorithms'):
        project_root = os.path.dirname(current_dir)
    else:
        project_root = current_dir

    INPUT_FOLDER = os.path.join(project_root, "dataset_crop")
    OUTPUT_FOLDER = os.path.join(project_root, "dataset_augmented")

    # Parameters
    GOOD_MULTIPLIER = 25
    DEFECTIVE_MULTIPLIER = 15
    SYNTHETIC_COUNT = 150

    # Verify folders exist
    good_folder = os.path.join(INPUT_FOLDER, "good")
    defective_folder = os.path.join(INPUT_FOLDER, "defective")

    if not os.path.exists(INPUT_FOLDER):
        return False

    if not os.path.exists(good_folder) and not os.path.exists(defective_folder):
        # Create folders automatically
        try:
            os.makedirs(good_folder, exist_ok=True)
            os.makedirs(defective_folder, exist_ok=True)
        except Exception:
            return False

    # Initialize augmentation system
    aug_system = CompleteAugmentationSystem(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER
    )

    # Execute pipeline
    success = aug_system.run_complete_augmentation(
        good_multiplier=GOOD_MULTIPLIER,
        defective_multiplier=DEFECTIVE_MULTIPLIER,
        synthetic_count=SYNTHETIC_COUNT
    )

    return success


if __name__ == "__main__":
    import sys

    # Execute main pipeline
    success = main_augmentation_pipeline()
    sys.exit(0 if success else 1)