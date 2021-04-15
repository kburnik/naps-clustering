"""Configuration for directories and constants."""

import os

DIR = os.path.dirname(os.path.abspath(__file__))
"""Root directory for the source code."""

DATA_DIR = os.path.join(DIR, '..', 'data')
"""Root directory for the data."""

OUT_DIR = os.path.join(DIR, '..', 'out')
"""Direktorij za rezultate."""

DATA_NAPS_ALL = os.path.join(DATA_DIR, 'NAPSA.csv')
"""Path to the NAPS dataset for clustering."""

DATA_NAPS_BE_ALL = os.path.join(DATA_DIR, 'emotion_and_stimuli.csv')
"""Path to the emotion and stimuli dataset for clustering."""
