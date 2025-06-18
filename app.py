"""
Hugging Face Spaces App for Cardboard Quality Control System
"""
import os
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main interface
from integrated_quality_interface import launch_integrated_interface

if __name__ == "__main__":
    # Launch the integrated interface
    launch_integrated_interface()