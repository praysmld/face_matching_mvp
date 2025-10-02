#!/usr/bin/env python3
"""
Hugging Face Spaces entry point for Face Matcher application
"""

import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from face_matcher.ui.gradio_app import main

if __name__ == "__main__":
    main()
