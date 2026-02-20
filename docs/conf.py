import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'Nexus'
copyright = '2026, Nexus Quant Team'
author = 'Nexus Quant Team'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
]

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
