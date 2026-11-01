"""Main entry point for AI Research Paper Analyzer"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.ui.app import main

if __name__ == "__main__":
    main()