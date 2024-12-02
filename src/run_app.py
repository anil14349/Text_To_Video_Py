import os
import sys
import streamlit as st
import logging
import warnings

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web.streamlit_app import main

# Configure logging
logging.basicConfig(level=logging.INFO)

# Suppress warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    main()
