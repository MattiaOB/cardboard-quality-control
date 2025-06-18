"""
Hugging Face Spaces App for Cardboard Quality Control System
"""

import os
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Import the main interface
    from integrated_quality_interface import launch_integrated_interface

    if __name__ == "__main__":
        # Launch the integrated interface
        launch_integrated_interface()

except Exception as e:
    print(f"Error launching application: {e}")
    import traceback

    traceback.print_exc()

    # Fallback: Create a simple error page
    import gradio as gr


    def error_interface():
        with gr.Blocks(title="Error") as demo:
            gr.Markdown(f"""
            # ❌ Application Error

            Sorry, there was an error loading the cardboard quality control system.

            **Error details:**
            ```
            {str(e)}
            ```

            Please check the logs or contact the developers.
            """)
        return demo


    error_app = error_interface()
    error_app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )