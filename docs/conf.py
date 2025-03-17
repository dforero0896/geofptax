import os
import sys
sys.path.insert(0, os.path.abspath('../'))

# Project information
project = 'GeoFPTax'
author = 'Daniel Forero-Sánchez'
release = '0.0.0'

# Extensions
extensions = [
    'sphinx.ext.autodoc',  # Automatically document modules
    'sphinx.ext.napoleon',  # Support for NumPy and Google docstrings
    'sphinx.ext.viewcode',  # Add links to source code
]

# HTML theme
html_theme = 'sphinx_rtd_theme'

# Paths
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']