"""
Cardboard Contour Detection Module
File: algorithms/contour_detection.py

FEATURES:
- Contour detection for cardboard using multiple strategies.
- Strategy 1: Adaptive threshold
- Strategy 2: Multiple fixed thresholds
- Strategy 3: Edge detection
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any


class CardboardContourDetector:
    """
    Cardboard contour detection using multiple strategies.
    """

    def __init__(self):
        """Initialize the contour detector."""
        pass

    def detect_cardboard_contour(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect cardboard contour using multiple strategies.
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Strategy 1: Adaptive threshold
        best_contour = self._try_adaptive_threshold(gray)
        if best_contour is not None:
            return best_contour

        # Strategy 2: Multiple fixed thresholds
        best_contour = self._try_multiple_thresholds(gray)
        if best_contour is not None:
            return best_contour

        # Strategy 3: Edge detection + morphology
        best_contour = self._try_edge_detection(gray)
        if best_contour is not None:
            return best_contour

        return None

    def _try_adaptive_threshold(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """Strategy 1: Adaptive threshold"""
        try:
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            adaptive = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

            # Try both normal and inverted
            for binary in [adaptive, cv2.bitwise_not(adaptive)]:
                contour = self._find_best_rectangular_contour(binary, gray.shape)
                if contour is not None:
                    return contour
            return None
        except Exception:
            return None

    def _try_multiple_thresholds(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """Strategy 2: Multiple fixed thresholds"""
        try:
            thresholds = [60, 80, 100, 120, 140, 160, 180, 200]

            for thresh_val in thresholds:
                _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
                _, binary_inv = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)

                for binary in [binary, binary_inv]:
                    # Morphological cleaning
                    kernel = np.ones((3, 3), np.uint8)
                    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
                    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

                    contour = self._find_best_rectangular_contour(cleaned, gray.shape)
                    if contour is not None:
                        return contour
            return None
        except Exception:
            return None

    def _try_edge_detection(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """Strategy 3: Edge detection"""
        try:
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            canny_params = [(50, 150), (30, 100), (100, 200)]

            for low, high in canny_params:
                edges = cv2.Canny(blurred, low, high)

                # Close gaps in edges
                kernel = np.ones((3, 3), np.uint8)
                edges = cv2.dilate(edges, kernel, iterations=2)
                edges = cv2.erode(edges, kernel, iterations=1)

                contour = self._find_best_rectangular_contour(edges, gray.shape)
                if contour is not None:
                    return contour
            return None
        except Exception:
            return None

    def _find_best_rectangular_contour(self, binary: np.ndarray, image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Find the best rectangular contour based on size, position and shape criteria"""
        h, w = image_shape
        image_area = h * w

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area (1% to 90% of image)
            if area < image_area * 0.01 or area > image_area * 0.9:
                continue

            x, y, width, height = cv2.boundingRect(contour)

            # Check not touching image borders
            margin = 5
            if (x < margin or y < margin or
                    x + width > w - margin or y + height > h - margin):
                continue

            # Check aspect ratio for cardboard
            aspect_ratio = width / height if height > 0 else 0
            if aspect_ratio < 0.1 or aspect_ratio > 10:
                continue

            # Calculate extent (contour area vs bounding box area)
            bbox_area = width * height
            extent = area / bbox_area if bbox_area > 0 else 0
            if extent < 0.3:
                continue

            # Calculate composite score
            area_score = min(area / (image_area * 0.5), 1.0)
            extent_score = extent
            position_score = self._calculate_position_score(x, y, width, height, w, h)
            total_score = area_score * 0.4 + extent_score * 0.4 + position_score * 0.2

            candidates.append({
                'contour': contour,
                'score': total_score
            })

        if not candidates:
            return None

        # Return best scoring contour
        best = max(candidates, key=lambda x: x['score'])
        return best['contour']

    def _calculate_position_score(self, x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> float:
        """Calculate score based on position (prefers centered objects)"""
        center_x = x + w // 2
        center_y = y + h // 2
        img_center_x = img_w // 2
        img_center_y = img_h // 2

        max_distance = np.sqrt((img_w / 2) ** 2 + (img_h / 2) ** 2)
        distance = np.sqrt((center_x - img_center_x) ** 2 + (center_y - img_center_y) ** 2)

        return 1.0 - (distance / max_distance)

    def create_cardboard_mask(self, image: np.ndarray, contour: Optional[np.ndarray],
                              padding_percent: float = 5.0) -> Optional[np.ndarray]:
        """
        Create binary mask of cardboard with optional internal padding.
        """
        if contour is None:
            return None

        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 255)

        # Apply erosion for internal padding
        if padding_percent > 0:
            padding_pixels = max(3, min(w, h) * padding_percent // 100)
            kernel = np.ones((int(padding_pixels), int(padding_pixels)), np.uint8)
            mask_eroded = cv2.erode(mask, kernel, iterations=1)

            # Check if eroded mask is still usable
            coverage = np.sum(mask_eroded > 0) / (h * w)
            if coverage < 0.02:  # Less than 2% coverage
                return mask
            return mask_eroded

        return mask

    def get_crop_info(self, contour: Optional[np.ndarray], image_shape: Tuple[int, int],
                      padding_percent: float = 10.0) -> Optional[Dict[str, Any]]:
        """
        Get cropping information for rectangular area containing cardboard.
        """
        if contour is None:
            return None

        h, w = image_shape[:2]
        x, y, bbox_w, bbox_h = cv2.boundingRect(contour)

        # Calculate padding
        padding_x = int(bbox_w * padding_percent / 100)
        padding_y = int(bbox_h * padding_percent / 100)

        # Expand rectangle with padding, keeping within image bounds
        x_new = max(0, x - padding_x)
        y_new = max(0, y - padding_y)
        w_new = min(w - x_new, bbox_w + 2 * padding_x)
        h_new = min(h - y_new, bbox_h + 2 * padding_y)

        return {
            'original_bbox': (x, y, bbox_w, bbox_h),
            'cropped_bbox': (x_new, y_new, w_new, h_new),
            'original_size': (w, h),
            'cropped_size': (w_new, h_new),
            'contour_area': cv2.contourArea(contour)
        }

    def crop_cardboard_area(self, image: np.ndarray, contour: Optional[np.ndarray],
                            padding_percent: float = 10.0) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """
        Crop rectangular area containing cardboard.
        """
        crop_info = self.get_crop_info(contour, image.shape, padding_percent)
        if crop_info is None:
            return None, None

        x_new, y_new, w_new, h_new = crop_info['cropped_bbox']
        cropped = image[y_new:y_new + h_new, x_new:x_new + w_new]

        if cropped.size == 0:
            return None, None

        return cropped, crop_info

    def create_safe_fallback_mask(self, image: np.ndarray, strategy: str = 'auto') -> np.ndarray:
        """
        Create fallback mask when contour detection fails.
        """
        h, w = image.shape[:2]

        if strategy == 'auto':
            try:
                # Convert to grayscale if needed
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image.copy()

                # Use Otsu for automatic threshold
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                binary_inv = cv2.bitwise_not(binary)

                # Choose version with reasonable coverage
                coverage_normal = np.sum(binary > 0) / (h * w)
                coverage_inv = np.sum(binary_inv > 0) / (h * w)

                if 0.1 <= coverage_normal <= 0.7:
                    chosen_binary = binary
                elif 0.1 <= coverage_inv <= 0.7:
                    chosen_binary = binary_inv
                else:
                    return self.create_safe_fallback_mask(image, 'center')

                # Morphological cleaning
                kernel = np.ones((5, 5), np.uint8)
                cleaned = cv2.morphologyEx(chosen_binary, cv2.MORPH_CLOSE, kernel, iterations=2)
                cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

                # Apply erosion for safety margin
                erode_kernel = np.ones((3, 3), np.uint8)
                return cv2.erode(cleaned, erode_kernel, iterations=1)

            except Exception:
                return self.create_safe_fallback_mask(image, 'center')

        elif strategy == 'center':
            # Central area fallback (50% of image dimensions)
            mask = np.zeros((h, w), dtype=np.uint8)
            margin_h = h // 4
            margin_w = w // 4
            mask[margin_h:h - margin_h, margin_w:w - margin_w] = 255
            return mask

        elif strategy == 'otsu':
            try:
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image.copy()
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                return binary
            except Exception:
                return self.create_safe_fallback_mask(image, 'center')

        else:
            # Unknown strategy, use center
            return self.create_safe_fallback_mask(image, 'center')