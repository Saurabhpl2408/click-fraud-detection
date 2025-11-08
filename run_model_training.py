#!/usr/bin/env python3
"""
Train fraud detection model
"""
import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.models.train import main

if __name__ == "__main__":
    main()
