"""
Hugging Face Spaces App for Cardboard Quality Control System
"""

import os
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("===== Application Startup =====")

try:
    # Try to import and launch the main interface
    print("Loading main interface...")
    from integrated_quality_interface import launch_integrated_interface

    launch_integrated_interface()

except ImportError as e:
    print(f"Import error: {e}")
    # Create a minimal working interface
    import gradio as gr

    with gr.Blocks(title="Import Error") as demo:
        gr.Markdown(f"""
        # ❌ Module Import Error

        Could not import required modules: {str(e)}

        This is likely due to missing dependencies or file structure issues.
        """)

    demo.launch(server_name="0.0.0.0", server_port=7860)

except Exception as e:
    print(f"Startup error: {e}")
    import traceback

    traceback.print_exc()

    # Create error interface
    import gradio as gr

    with gr.Blocks(title="Startup Error") as demo:
        gr.Markdown(f"""
        # ⚠️ Application Startup Error

        **Error details:**
        ```
        {str(e)}
        ```

        **Project Info:**
        - Course: Advanced Measurement Systems for Control Applications
        - Institution: Politecnico di Milano
        - Students: Mattia Ogliar Badessi, Luca Molettieri

        Please check the logs for more details.
        """)

    demo.launch(server_name="0.0.0.0", server_port=7860)