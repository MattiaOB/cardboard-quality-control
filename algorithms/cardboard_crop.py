"""
Cardboard Real Detection and Cropping Algorithm
File: algorithms/cardboard_crop.py

Features:
1. Detect real cardboard contour (any orientation)
2. Find a rectangle that completely contains it
3. Crop only that rectangular area (without resize)
4. Save cropped area in original dimensions
"""

import cv2
import numpy as np
import os
import sys
from typing import Optional, Tuple

# Import the unified contour detection module
try:
    from .contour_detection import CardboardContourDetector
except ImportError:
    # Fallback for direct execution
    from contour_detection import CardboardContourDetector


class RealCardboardDetector:
    """
    Real cardboard detector with precise cropping.
    Uses unified contour detection module.
    """

    def __init__(self):
        """Initialize the detector."""
        self.setup_paths()
        self.contour_detector = CardboardContourDetector()

    def setup_paths(self):
        """Setup input and output directory paths"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)

        self.source_path = os.path.join(project_root, "dataset_original")
        self.output_path = os.path.join(project_root, "dataset_crop")

        # Create output folders
        os.makedirs(self.output_path, exist_ok=True)
        for category in ['good', 'defective']:
            os.makedirs(os.path.join(self.output_path, category), exist_ok=True)

    def process_image(self, input_path: str, output_path: str) -> bool:
        """
        Process single image using unified contour detection.

        Args:
            input_path: Path to input image
            output_path: Path to save cropped image

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load image
            image = cv2.imread(input_path)
            if image is None:
                return False

            # Detect cardboard contour using unified detector
            contour = self.contour_detector.detect_cardboard_contour(image)
            if contour is None:
                return False

            # Crop cardboard area with padding
            cropped, crop_info = self.contour_detector.crop_cardboard_area(
                image, contour, padding_percent=15
            )

            if cropped is None or cropped.size == 0:
                return False

            # Save cropped image
            return cv2.imwrite(output_path, cropped)

        except Exception:
            return False

    def process_category(self, category: str) -> int:
        """
        Process all images in a category folder.

        Args:
            category: Category name ('good' or 'defective')

        Returns:
            Number of successfully processed images
        """
        category_input = os.path.join(self.source_path, category)
        category_output = os.path.join(self.output_path, category)

        if not os.path.exists(category_input):
            return 0

        # Find all images
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = [f for f in os.listdir(category_input)
                      if f.lower().endswith(image_extensions)]

        if not image_files:
            return 0

        processed_count = 0
        for filename in image_files:
            input_path = os.path.join(category_input, filename)
            output_path = os.path.join(category_output, filename)

            if self.process_image(input_path, output_path):
                processed_count += 1

        return processed_count

    def process_all(self) -> bool:
        """
        Process entire dataset.

        Returns:
            True if any images were processed successfully
        """
        if not os.path.exists(self.source_path):
            print(f"Source path not found: {self.source_path}")
            return False

        print("Starting cardboard auto-crop...")

        total_processed = 0
        for category in ['good', 'defective']:
            category_processed = self.process_category(category)
            total_processed += category_processed

            if category_processed > 0:
                print(f"{category}: {category_processed} images processed")

        print(f"Total images processed: {total_processed}")
        return total_processed > 0

    def create_debug_visualization(self, image_path: str, save_debug: bool = True) -> bool:
        """
        Create debug visualization for a single image.

        Args:
            image_path: Path to test image
            save_debug: Whether to save debug images

        Returns:
            True if successful
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return False

            # Detect contour
            contour = self.contour_detector.detect_cardboard_contour(image)
            if contour is None:
                return False

            if save_debug:
                debug_folder = "debug_auto_crop"
                os.makedirs(debug_folder, exist_ok=True)

                # Save original
                cv2.imwrite(os.path.join(debug_folder, "01_original.jpg"), image)

                # Create and save mask
                mask = self.contour_detector.create_cardboard_mask(image, contour, padding_percent=5)
                if mask is not None:
                    cv2.imwrite(os.path.join(debug_folder, "02_mask.jpg"), mask)

                # Create overlay with contour
                overlay = image.copy()
                if mask is not None:
                    overlay[mask == 0] = overlay[mask == 0] * 0.3
                cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 3)
                cv2.imwrite(os.path.join(debug_folder, "03_overlay.jpg"), overlay)

                # Save cropped result
                cropped, _ = self.contour_detector.crop_cardboard_area(
                    image, contour, padding_percent=15
                )
                if cropped is not None:
                    cv2.imwrite(os.path.join(debug_folder, "04_cropped.jpg"), cropped)

            return True

        except Exception:
            return False


def main():
    """Main function with command line arguments"""
    import argparse

    parser = argparse.ArgumentParser(description='Cardboard Auto-Crop')
    parser.add_argument('--visualize', type=str, help='Create debug visualization for specific image')
    parser.add_argument('--test', action='store_true', help='Test mode with single image')

    args = parser.parse_args()

    # Initialize detector
    detector = RealCardboardDetector()

    if args.visualize:
        # Create debug visualization
        success = detector.create_debug_visualization(args.visualize, save_debug=True)
        return success

    elif args.test:
        # Test mode - find and process single image
        test_folders = ["dataset_original/good", "dataset_original/defective", "test_images", "."]
        test_image = None

        for folder in test_folders:
            if os.path.exists(folder):
                for file in os.listdir(folder):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        test_image = os.path.join(folder, file)
                        break
                if test_image:
                    break

        if test_image:
            success = detector.create_debug_visualization(test_image, save_debug=True)
            return success
        else:
            print("No test image found")
            return False

    else:
        # Normal processing
        success = detector.process_all()
        return success


if __name__ == "__main__":
    success = main()

    sys.exit(0 if success else 1)