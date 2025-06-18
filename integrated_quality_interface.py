"""
Integrated Gradio Interface for Cardboard Quality Control
File: integrated_quality_interface.py

Features:
1. Contour Check: Contour and cut analysis
2. Surface Check: Surface defect detection with Few-Shot model
3. Integrated image utilities (no external dependencies)
"""

import gradio as gr
import cv2
import numpy as np
from PIL import Image
import os
import sys
import glob
from typing import List, Tuple, Optional

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
sys.path.append(project_root)

# Import required modules
try:
    from algorithms.contour_check import ContourChecker
    from algorithms.few_shot_trainer import FewShotPredictor
    from models.model_config import CartonModelConfig
except ImportError as e:
    print(f"Import error: {e}")


class IntegratedQualityInterface:
    """Integrated interface for cardboard quality control"""

    def __init__(self):
        """Initialize the interface"""
        self.setup_paths()
        self.model_config_manager = CartonModelConfig()
        self.surface_predictor = None

        # Load surface model
        self.load_surface_model()

        # Get available test images
        self.contour_test_images = self.get_test_images("contour")
        self.surface_test_images = self.get_test_images("surface")

        # Current image indices for navigation
        self.current_contour_index = 0
        self.current_surface_index = 0

    @staticmethod
    def pil_to_cv2(pil_image):
        """Convert from PIL to OpenCV format"""
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    @staticmethod
    def cv2_to_pil(cv2_image):
        """Convert from OpenCV to PIL format"""
        return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

    def setup_paths(self):
        """Setup project paths"""
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.test_images_root = os.path.join(self.project_root, "test_images")
        self.models_dir = os.path.join(self.project_root, "models")
        self.golden_samples_dir = os.path.join(self.project_root, "golden_samples")

    def get_test_images(self, category: str) -> List[Tuple[str, str]]:
        """Get list of test images for a category"""
        test_dir = os.path.join(self.test_images_root, category)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = set()

        all_files = os.listdir(test_dir)

        for filename in all_files:
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in image_extensions:
                full_path = os.path.join(test_dir, filename)
                if os.path.isfile(full_path):
                    image_files.add(full_path)

        # Convert set to sorted list and return tuples of (full_path, filename)
        sorted_files = sorted(list(image_files))
        return [(f, os.path.basename(f)) for f in sorted_files]

    def get_current_contour_image(self) -> Tuple[str, str]:
        """Get current contour image path and info"""
        if self.current_contour_index >= len(self.contour_test_images):
            self.current_contour_index = 0

        img_path, filename = self.contour_test_images[self.current_contour_index]
        info = f"Image {self.current_contour_index + 1} of {len(self.contour_test_images)}"
        return img_path, info

    def get_current_surface_image(self) -> Tuple[str, str]:
        """Get current surface image path and info"""
        if self.current_surface_index >= len(self.surface_test_images):
            self.current_surface_index = 0

        img_path, filename = self.surface_test_images[self.current_surface_index]
        info = f"Image {self.current_surface_index + 1} of {len(self.surface_test_images)}"
        return img_path, info

    def navigate_contour_images(self, direction: str) -> Tuple[str, str]:
        """Navigate through contour images"""
        if direction == "next":
            self.current_contour_index = (self.current_contour_index + 1) % len(self.contour_test_images)
        elif direction == "prev":
            self.current_contour_index = (self.current_contour_index - 1) % len(self.contour_test_images)

        return self.get_current_contour_image()

    def navigate_surface_images(self, direction: str) -> Tuple[str, str]:
        """Navigate through surface images"""
        if direction == "next":
            self.current_surface_index = (self.current_surface_index + 1) % len(self.surface_test_images)
        elif direction == "prev":
            self.current_surface_index = (self.current_surface_index - 1) % len(self.surface_test_images)

        return self.get_current_surface_image()

    def load_surface_model(self):
        """Load the Few-Shot surface defect detection model"""
        # Find the latest model file
        model_pattern = os.path.join(self.models_dir, "few_shot_carton_model_*.pt")
        model_files = glob.glob(model_pattern)

        # Use the most recent model
        latest_model = max(model_files, key=os.path.getmtime)
        self.surface_predictor = FewShotPredictor(latest_model, device='auto')

        return self.surface_predictor.is_loaded

    def process_contour_check(self) -> Tuple[str, Optional[Image.Image]]:
        """Process contour check on current selected test image"""
        img_path, _ = self.get_current_contour_image()

        test_image_path = img_path
        selected_image_name = os.path.basename(img_path)
        golden_sample_path = os.path.join(self.golden_samples_dir, "small", "contour_template.jpg")

        # Get model config
        model_config = self.model_config_manager.get_model_config("Small")

        # Initialize contour checker and perform check
        contour_checker = ContourChecker(model_config)
        result = contour_checker.check_contours(test_image_path, golden_sample_path)

        # Format result text
        result_text = self._format_contour_result_text(result, selected_image_name)

        # Convert result image
        result_image = None
        if result['result_image'] is not None:
            result_image = self.cv2_to_pil(result['result_image'])

        return result_text, result_image

    def process_surface_check(self) -> Tuple[str, Optional[Image.Image]]:
        """Process surface defect check on current selected test image"""
        img_path, _ = self.get_current_surface_image()

        test_image_path = img_path
        selected_image_name = os.path.basename(img_path)

        # Load and process image
        test_image = cv2.imread(test_image_path)

        # Perform prediction
        prediction_result = self.surface_predictor.predict(test_image, return_confidence=True)

        # Format result text and create visualization
        result_text = self._format_surface_result_text(prediction_result, selected_image_name)
        result_image = self._create_surface_visualization(test_image, prediction_result)

        return result_text, result_image

    def _format_contour_result_text(self, result: dict, image_name: str) -> str:
        """Format contour check result text"""
        status = "✅ PASSED" if result['passed'] else "❌ FAILED"

        result_text = f"""

## Status: {status}

### Result Message:
{result['message']}

        """

        return result_text.strip()

    def _format_surface_result_text(self, result: dict, image_name: str) -> str:
        """Format surface check result text"""
        is_good = result['class'] == 0  # 0 = good, 1 = defective
        status = "✅ PASSED" if is_good else "❌ FAILED"

        result_text = f"""

## Status: {status}

### AI Model Prediction:
- **Classification**: {result['class_name'].upper()}
- **Confidence**: {result['confidence']:.3f} ({result['confidence']*100:.1f}%)
- **Inference Time**: {result['inference_time_ms']:.1f}ms

        """

        return result_text.strip()

    def _create_surface_visualization(self, original_image: np.ndarray, prediction: dict) -> Image.Image:
        """Create visualization for surface check results"""
        # Create a copy for visualization
        vis_image = original_image.copy()
        height, width = vis_image.shape[:2]

        # Determine colors based on prediction
        is_good = prediction['class'] == 0
        border_color = (0, 255, 0) if is_good else (0, 0, 255)
        text_color = (0, 255, 0) if is_good else (0, 0, 255)

        # Add border
        border_thickness = max(5, min(width, height) // 100)
        cv2.rectangle(vis_image, (0, 0), (width-1, height-1), border_color, border_thickness)

        # Prepare text overlay
        class_text = prediction['class_name'].upper()
        confidence_text = f"{prediction['confidence']:.3f}"
        decision_text = prediction['decision']

        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.7, min(width, height) / 800)
        thickness = max(2, int(font_scale * 2))

        # Background rectangle for text
        overlay = vis_image.copy()
        cv2.rectangle(overlay, (10, 10), (min(400, width-20), 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, vis_image, 0.3, 0, vis_image)

        # Add text
        y_offset = 35
        cv2.putText(vis_image, f"Class: {class_text}", (20, y_offset),
                   font, font_scale, text_color, thickness)

        y_offset += 30
        cv2.putText(vis_image, f"Confidence: {confidence_text}", (20, y_offset),
                   font, font_scale, text_color, thickness)

        y_offset += 30
        cv2.putText(vis_image, f"Decision: {decision_text}", (20, y_offset),
                   font, font_scale, text_color, thickness)

        # Convert to PIL Image
        vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(vis_image_rgb)

        return pil_image

    def get_model_info(self) -> str:
        """Get information about loaded models"""
        info_text = "# 📊 Model Information\n\n"

        # Contour model info
        info_text += "## 🔍 Contour Detection Model\n"
        info_text += "- **Type**: Rule-based contour analysis\n"
        info_text += "- **Golden Sample**: golden_samples/small/contour_template.jpg\n"
        info_text += "- **Features**: Vertex detection, defect analysis, shape matching\n\n"

        # Surface model info
        info_text += "## 🌡️ Surface Defect Detection Model\n"
        model_info = self.surface_predictor.get_model_info()
        info_text += f"- **Type**: {model_info.get('model_type', 'N/A')}\n"
        info_text += f"- **Architecture**: {model_info.get('model_name', 'N/A')}\n"
        info_text += f"- **Classes**: {', '.join(model_info.get('class_names', ['N/A']))}\n"
        info_text += f"- **Training Accuracy**: {model_info.get('training_accuracy', 'N/A')}\n"
        info_text += f"- **Parameters**: {model_info.get('parameters', 'N/A'):,}\n"
        info_text += f"- **Model Size**: {model_info.get('model_size_mb', 0):.1f} MB\n"
        info_text += f"- **Device**: {model_info.get('device', 'N/A')}\n"
        info_text += f"- **Timestamp**: {model_info.get('timestamp', 'N/A')}\n"

        return info_text

    def refresh_test_images(self) -> Tuple[str, str, str, str]:
        """Refresh the list of available test images"""
        self.contour_test_images = self.get_test_images("contour")
        self.surface_test_images = self.get_test_images("surface")

        # Reset indices
        self.current_contour_index = 0
        self.current_surface_index = 0

        # Get current images
        contour_img, contour_info = self.get_current_contour_image()
        surface_img, surface_info = self.get_current_surface_image()

        return contour_img, contour_info, surface_img, surface_info

    def create_interface(self):
        """Create the Gradio interface"""
        with gr.Blocks(title="Demo Cardboard Quality Control", theme=gr.themes.Soft()) as interface:

            # Header Section
            with gr.Row():
                with gr.Column(scale=1):
                    # University Logo
                    logo_path = os.path.join(self.project_root, "logo.svg")
                    if os.path.exists(logo_path):
                        gr.Image(
                            value=logo_path,
                            height=120,
                            width=250,
                            show_label=False,
                            show_download_button=False,
                            container=False,
                            interactive=False
                        )
                    else:
                        gr.Markdown("🎓 **POLIMI**")

                with gr.Column(scale=3):
                    gr.Markdown(
                        """
                        # Demo Cardboard Quality Control System
                        ### Course: Advanced Measurement Systems for Control Applications
                        ### Students: Mattia Ogliar Badessi, Luca Molettieri
                        """,
                        elem_classes="university-title"
                    )

            gr.Markdown("---")
            gr.Markdown("Comprehensive quality assessment with contour analysis and surface defect detection")

            with gr.Tabs():
                # Tab 1: Contour Check
                with gr.TabItem("🔍 Contour Quality Check"):
                    gr.Markdown("## Contour and Cut Quality Analysis")
                    gr.Markdown("Analyzes cardboard contours, vertices, and internal cuts against golden sample")

                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### 📸 Test Images:")

                            # Navigation controls
                            with gr.Row():
                                contour_prev_btn = gr.Button("⬅️ Previous", size="sm")
                                contour_next_btn = gr.Button("➡️ Next", size="sm")

                            # Current image display
                            contour_current_image = gr.Image(
                                label="Current Test Image",
                                height=250,
                                interactive=False,
                                value=self.get_current_contour_image()[0] if self.contour_test_images else None
                            )

                            # Image counter
                            contour_image_info = gr.Markdown(
                                value=self.get_current_contour_image()[1] if self.contour_test_images else "No images",
                                elem_classes="image-counter"
                            )

                            contour_analyze_btn = gr.Button(
                                "Analyze Contours",
                                variant="primary",
                                size="lg"
                            )

                        with gr.Column(scale=2):
                            contour_result_text = gr.Markdown(
                                "Navigate through test images and click 'Analyze Contours' to start",
                                label="Analysis Results"
                            )

                            contour_result_image = gr.Image(
                                label="Contour Analysis Visualization",
                                height=400
                            )

                # Tab 2: Surface Check
                with gr.TabItem("🌡️ Surface Quality Check"):
                    gr.Markdown("## Surface Defect Detection")
                    gr.Markdown("AI-powered surface defect detection using Few-Shot Learning")

                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### 📸 Test Images:")

                            # Navigation controls
                            with gr.Row():
                                surface_prev_btn = gr.Button("⬅️ Previous", size="sm")
                                surface_next_btn = gr.Button("➡️ Next", size="sm")

                            # Current image display
                            surface_current_image = gr.Image(
                                label="Current Test Image",
                                height=250,
                                interactive=False,
                                value=self.get_current_surface_image()[0] if self.surface_test_images else None
                            )

                            # Image counter
                            surface_image_info = gr.Markdown(
                                value=self.get_current_surface_image()[1] if self.surface_test_images else "No images",
                                elem_classes="image-counter"
                            )

                            surface_analyze_btn = gr.Button(
                                "Analyze Surface",
                                variant="primary",
                                size="lg"
                            )

                        with gr.Column(scale=2):
                            surface_result_text = gr.Markdown(
                                "Navigate through test images and click 'Analyze Surface' to start",
                                label="Analysis Results"
                            )

                            surface_result_image = gr.Image(
                                label="Surface Analysis Visualization",
                                height=400
                            )

                # Tab 3: Model Information
                with gr.TabItem("📊 System Information"):
                    model_info_text = gr.Markdown(
                        self.get_model_info(),
                        label="Model Information"
                    )

                    refresh_btn = gr.Button("🔄 Refresh Images List", variant="secondary")

                    with gr.Accordion("📁 Directory Structure", open=False):
                        gr.Markdown(f"""                      
                        ### Usage Instructions:
                        1. **Contour Check**: Place test images in `test_images/contour/`
                        2. **Surface Check**: Place test images in `test_images/surface/`
                        3. Golden sample: `golden_samples/small/contour_template.jpg`
                        4. Trained model: `models/few_shot_carton_model_*.pt`
                        """)

            # Event handlers
            def update_contour_navigation(direction):
                img_path, info = self.navigate_contour_images(direction)
                return gr.Image(value=img_path), info

            def update_surface_navigation(direction):
                img_path, info = self.navigate_surface_images(direction)
                return gr.Image(value=img_path), info

            # Contour navigation
            contour_prev_btn.click(
                fn=lambda: update_contour_navigation("prev"),
                outputs=[contour_current_image, contour_image_info]
            )

            contour_next_btn.click(
                fn=lambda: update_contour_navigation("next"),
                outputs=[contour_current_image, contour_image_info]
            )

            # Surface navigation
            surface_prev_btn.click(
                fn=lambda: update_surface_navigation("prev"),
                outputs=[surface_current_image, surface_image_info]
            )

            surface_next_btn.click(
                fn=lambda: update_surface_navigation("next"),
                outputs=[surface_current_image, surface_image_info]
            )

            # Analysis event handlers
            contour_analyze_btn.click(
                fn=self.process_contour_check,
                outputs=[contour_result_text, contour_result_image]
            )

            surface_analyze_btn.click(
                fn=self.process_surface_check,
                outputs=[surface_result_text, surface_result_image]
            )

            # Refresh functionality
            def refresh_all_images():
                contour_img, contour_info, surface_img, surface_info = self.refresh_test_images()
                return (
                    gr.Image(value=contour_img),
                    contour_info,
                    gr.Image(value=surface_img),
                    surface_info
                )

            refresh_btn.click(
                fn=refresh_all_images,
                outputs=[contour_current_image, contour_image_info, surface_current_image, surface_image_info]
            )

        return interface


def launch_integrated_interface():
    """Launch the integrated interface for Hugging Face Spaces"""
    import os

    try:
        # Initialize interface
        interface_manager = IntegratedQualityInterface()

        # Create interface
        interface = interface_manager.create_interface()

        # Simplified launch configuration for HF Spaces
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True,
            share=False,
            quiet=False
        )

    except Exception as e:
        print(f"Error launching main interface: {e}")
        import traceback
        traceback.print_exc()

        # Create fallback interface
        import gradio as gr

        def create_fallback_interface():
            with gr.Blocks(title="Demo Unavailable") as demo:

                with gr.Row():
                    gr.Image(
                        "https://via.placeholder.com/400x300/blue/white?text=Cardboard+QC+Demo",
                        label="Demo Preview",
                        interactive=False
                    )
            return demo

        fallback_app = create_fallback_interface()
        fallback_app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True
        )


if __name__ == "__main__":
    launch_integrated_interface()