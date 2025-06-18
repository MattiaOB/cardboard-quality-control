"""
Cardboard Contour Detection Algorithm
File: algorithms/contour_check.py

FEATURES:
- Detects vertices based on abrupt changes in the contour slope
- Modified to remove red and orange point visualization
"""
import cv2
import numpy as np
from PIL import Image


class ContourChecker:
    def __init__(self, model_config):
        self.model_config = model_config

    @staticmethod
    def load_image(image_path):
        """Load an image from file"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot load image: {image_path}")
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            return None

    @staticmethod
    def cv2_to_pil(cv2_image):
        """Convert from OpenCV to PIL format"""
        return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

    def check_contours(self, input_image_path, golden_template_path):
        """Main contour checking function with vertex detection"""
        try:
            # Load images
            input_image = self.load_image(input_image_path)
            golden_template = self.load_image(golden_template_path)

            if input_image is None or golden_template is None:
                return self._create_error_result("Error loading images")

            # Analyze both images
            input_analysis = self._analyze_cardboard(input_image)
            template_analysis = self._analyze_cardboard(golden_template)

            # Compare results
            comparison_result = self._compare_cardboards(input_analysis, template_analysis)

            # Create result image
            result_image = self._create_result_image(input_image, input_analysis, comparison_result)

            # Determine pass/fail
            threshold = self.model_config['contour_params']['template_matching_threshold']
            is_passed = comparison_result['overall_similarity'] >= threshold

            return {
                'passed': is_passed,
                'similarity_score': comparison_result['overall_similarity'],
                'detected_contours': input_analysis['total_features'],
                'expected_contours': template_analysis['total_features'],
                'missing_contours': comparison_result.get('missing_features', 0),
                'extra_contours': comparison_result.get('extra_features', 0),
                'result_image': result_image,
                'message': self._generate_message(is_passed, comparison_result)
            }

        except Exception as e:
            return self._create_error_result(f"Analysis error: {str(e)}")

    def _analyze_cardboard(self, image):
        """Analyze cardboard with vertex detection"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        height, width = gray.shape

        # Preprocessing
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # External contour detection
        _, mask_cardboard = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)

        # Morphological cleaning
        kernel = np.ones((5, 5), np.uint8)
        mask_cardboard = cv2.morphologyEx(mask_cardboard, cv2.MORPH_CLOSE, kernel)
        mask_cardboard = cv2.morphologyEx(mask_cardboard, cv2.MORPH_OPEN, kernel)

        # Find main contour
        external_contours, _ = cv2.findContours(mask_cardboard, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        main_contour = None
        if external_contours:
            areas = [cv2.contourArea(c) for c in external_contours]
            max_area_idx = np.argmax(areas)
            if areas[max_area_idx] > 10000:
                main_contour = external_contours[max_area_idx]

        # Detect vertices and defects
        main_vertices, small_defects = self._detect_vertices_and_defects(main_contour)

        # Internal holes detection
        _, mask_holes = cv2.threshold(blurred, 160, 255, cv2.THRESH_BINARY)

        internal_holes = []
        if main_contour is not None:
            # Create mask for internal area
            contour_mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillPoly(contour_mask, [main_contour], 255)
            eroded_mask = cv2.erode(contour_mask, np.ones((15, 15), np.uint8), iterations=1)
            mask_holes_internal = cv2.bitwise_and(mask_holes, eroded_mask)

            hole_contours, _ = cv2.findContours(mask_holes_internal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter holes by size and position
            for contour in hole_contours:
                area = cv2.contourArea(contour)
                if 800 <= area <= 15000:
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x, center_y = x + w//2, y + h//2
                    margin = min(width, height) // 6
                    if (margin < center_x < width - margin and margin < center_y < height - margin):
                        internal_holes.append(contour)

        # Shape features
        shape_features = {}
        if main_contour is not None:
            area = cv2.contourArea(main_contour)
            perimeter = cv2.arcLength(main_contour, True)
            shape_features = {'area': area, 'perimeter': perimeter}

        return {
            'main_contour': main_contour,
            'internal_holes': internal_holes,
            'main_vertices': main_vertices,
            'small_defects': small_defects,
            'shape_features': shape_features,
            'total_features': len(internal_holes) + (1 if main_contour is not None else 0),
            'holes_count': len(internal_holes),
            'vertices_count': len(main_vertices),
            'defects_count': len(small_defects),
            'main_area': shape_features.get('area', 0)
        }

    def _detect_vertices_and_defects(self, contour):
        """Detect main vertices and small defects"""
        if contour is None or len(contour) < 20:
            return [], []

        try:
            contour_points = contour.reshape(-1, 2)
            main_vertices = self._detect_main_vertices(contour_points)
            small_defects = self._detect_small_contour_defects(contour_points)
            return main_vertices, small_defects
        except Exception:
            return [], []

    def _detect_main_vertices(self, contour_points):
        """Detect main vertices based on angle changes"""
        vertices = []
        window_size = 15
        angle_threshold = 25
        min_distance = 30

        # Calculate angles
        angles = []
        for i in range(len(contour_points)):
            prev_idx = (i - window_size) % len(contour_points)
            next_idx = (i + window_size) % len(contour_points)

            prev_point = contour_points[prev_idx]
            curr_point = contour_points[i]
            next_point = contour_points[next_idx]

            v1 = prev_point - curr_point
            v2 = next_point - curr_point

            angle = self._calculate_angle(v1, v2)
            angles.append(angle)

        # Find abrupt changes
        for i in range(len(angles)):
            prev_angle = angles[i-5] if i >= 5 else angles[i]
            next_angle = angles[(i+5) % len(angles)]
            current_angle = angles[i]

            angle_change = abs(current_angle - prev_angle) + abs(current_angle - next_angle)

            if angle_change > angle_threshold:
                vertex_point = contour_points[i]

                # Check minimum distance from other vertices
                too_close = any(np.linalg.norm(vertex_point - existing_vertex) < min_distance
                               for existing_vertex in vertices)

                if not too_close:
                    vertices.append(vertex_point)

        return vertices

    def _detect_small_contour_defects(self, contour_points):
        """Detect small cuts/defects in contour"""
        defects = []

        try:
            smoothing_window = 8
            deviation_threshold = 3.0
            min_defect_distance = 15

            # Create smooth contour using moving average
            smooth_contour = []
            for i in range(len(contour_points)):
                points_sum = np.array([0.0, 0.0])
                count = 0

                for j in range(-smoothing_window, smoothing_window + 1):
                    idx = (i + j) % len(contour_points)
                    points_sum += contour_points[idx]
                    count += 1

                smooth_point = points_sum / count
                smooth_contour.append(smooth_point)

            smooth_contour = np.array(smooth_contour)

            # Calculate deviation from smooth contour
            for i in range(len(contour_points)):
                original_point = contour_points[i]
                smooth_point = smooth_contour[i]

                deviation = np.linalg.norm(original_point - smooth_point)

                if deviation > deviation_threshold:
                    # Check minimum distance from other defects
                    too_close = any(np.linalg.norm(original_point - existing_defect) < min_defect_distance
                                   for existing_defect in defects)

                    if not too_close:
                        defects.append(original_point)

            return defects

        except Exception:
            return []

    def _calculate_angle(self, v1, v2):
        """Calculate angle between two vectors in degrees"""
        try:
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 == 0 or norm2 == 0:
                return 0

            v1_norm = v1 / norm1
            v2_norm = v2 / norm2

            cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
            angle = np.arccos(cos_angle) * 180 / np.pi

            return angle
        except:
            return 0

    def _compare_cardboards(self, input_analysis, template_analysis):
        """Compare analysis results"""
        comparison = {
            'contour_similarity': 0.0,
            'holes_similarity': 0.0,
            'vertices_similarity': 0.0,
            'defects_similarity': 0.0,
            'overall_similarity': 0.0,
            'issues': []
        }

        # Compare hole count
        input_holes = input_analysis['holes_count']
        template_holes = template_analysis['holes_count']

        if input_holes == template_holes:
            holes_similarity = 1.0
        else:
            holes_similarity = 0.0
            comparison['issues'].append(f"Different holes: {input_holes} vs {template_holes}")

        comparison['holes_similarity'] = holes_similarity

        # Compare main contour
        if (input_analysis['main_contour'] is not None and template_analysis['main_contour'] is not None):
            try:
                shape_distance = cv2.matchShapes(
                    input_analysis['main_contour'],
                    template_analysis['main_contour'],
                    cv2.CONTOURS_MATCH_I1, 0
                )
                contour_similarity = max(0.5, 1.0 - min(shape_distance, 0.5))
            except:
                contour_similarity = 0.7
        else:
            contour_similarity = 0.0

        comparison['contour_similarity'] = contour_similarity

        # Compare vertices count
        input_vertices = input_analysis['vertices_count']
        template_vertices = template_analysis['vertices_count']

        if input_vertices == template_vertices:
            vertices_similarity = 1.0
        else:
            vertex_diff = abs(input_vertices - template_vertices)
            if vertex_diff <= 1:
                vertices_similarity = 0.8
            else:
                vertices_similarity = 0.3
                comparison['issues'].append(f"Very different vertices: {input_vertices} vs {template_vertices}")

        comparison['vertices_similarity'] = vertices_similarity

        # Compare small defects
        input_defects = input_analysis['defects_count']
        template_defects = template_analysis['defects_count']

        if input_defects <= template_defects + 2:
            defects_similarity = 1.0 - (input_defects - template_defects) * 0.1
            defects_similarity = max(0.0, defects_similarity)
        else:
            defects_similarity = 0.0
            comparison['issues'].append(f"Too many contour defects: {input_defects} vs {template_defects}")

        comparison['defects_similarity'] = defects_similarity

        # Calculate final score
        if input_holes != template_holes:
            comparison['overall_similarity'] = 0.3
        else:
            comparison['overall_similarity'] = (
                holes_similarity * 0.4 +      # 40% holes
                contour_similarity * 0.25 +   # 25% contour
                vertices_similarity * 0.2 +   # 20% vertices
                defects_similarity * 0.15     # 15% small defects
            )

        return comparison

    def _create_result_image(self, original_image, analysis, comparison):
        """Create result image with highlighted elements - WITHOUT red and orange points"""
        result = original_image.copy()

        # Draw main contour in green
        if analysis['main_contour'] is not None:
            cv2.drawContours(result, [analysis['main_contour']], -1, (0, 255, 0), 4)

        # REMOVED: Main vertices drawing (red points)
        # REMOVED: Small defects drawing (orange points)

        # Draw holes in blue with numbering
        for i, hole in enumerate(analysis['internal_holes']):
            cv2.drawContours(result, [hole], -1, (255, 0, 0), 2)
            M = cv2.moments(hole)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(result, str(i+1), (cx-10, cy+10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)

        # Information overlay
        overlay = result.copy()
        cv2.rectangle(overlay, (10, 10), (450, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, result, 0.2, 0, result)

        font = cv2.FONT_HERSHEY_SIMPLEX
        vertices_similarity = comparison.get('vertices_similarity', 0.0)
        defects_similarity = comparison.get('defects_similarity', 0.0)

        info_lines = [
            f"Similarity: {comparison['overall_similarity']:.3f}",
            f"Contour: {comparison['contour_similarity']:.3f}",
            f"Holes: {comparison['holes_similarity']:.3f}",
            f"Vertices: {vertices_similarity:.3f}",
            f"Defects: {defects_similarity:.3f}",
            f"Vertices: {analysis['vertices_count']} | Defects: {analysis['defects_count']}",
            f"Holes found: {analysis['holes_count']}"
        ]

        for i, line in enumerate(info_lines):
            color = (255, 255, 255)
            if i == 0:  # Overall similarity
                color = (0, 255, 0) if comparison['overall_similarity'] > 0.7 else (0, 0, 255)
            elif i == 3:  # Vertices
                color = (0, 255, 0) if vertices_similarity > 0.8 else (255, 255, 0) if vertices_similarity > 0.6 else (0, 0, 255)
            elif i == 4:  # Defects
                color = (0, 255, 0) if defects_similarity > 0.8 else (255, 255, 0) if defects_similarity > 0.6 else (0, 0, 255)

            cv2.putText(result, line, (15, 35 + i*21), font, 0.65, color, 2)

        return result

    def _generate_message(self, is_passed, comparison_result):
        """Generate result message"""
        if is_passed:
            return f"Check PASSED. Cardboard compliant (similarity: {comparison_result['overall_similarity']:.3f})"
        else:
            issues = comparison_result.get('issues', [])
            if issues:
                return f"Check FAILED. {issues[0]}. Similarity: {comparison_result['overall_similarity']:.3f}"
            else:
                return f"Check FAILED. Insufficient similarity: {comparison_result['overall_similarity']:.3f}"

    def _create_error_result(self, error_message):
        """Create error result"""
        return {
            'passed': False,
            'similarity_score': 0.0,
            'detected_contours': 0,
            'expected_contours': 0,
            'missing_contours': 0,
            'extra_contours': 0,
            'result_image': None,
            'message': error_message
        }