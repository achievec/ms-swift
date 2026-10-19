# Copyright (c) ModelScope Contributors. All rights reserved.
"""Recursive self-improvement loop on top of ms-swift.

This package does not import ms-swift. Training, eval, and deploy go through
MsSwiftBackend by default (``python -m swift.cli.main`` on GPU 0). Set
``RSI_BACKEND=mock`` to click through the loop without a GPU.
"""

__version__ = '0.1.0'
