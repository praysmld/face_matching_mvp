#!/usr/bin/env python3
"""
Convenience script to run the Face Matcher Gradio application
"""

import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from face_matcher.ui.gradio_app import main

if __name__ == "__main__":
    main()
