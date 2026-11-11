# Copyright (c) ModelScope Contributors. All rights reserved.
"""Recursive self-improvement loop on top of ms-swift.

This package does not import ms-swift. Training and evaluation go through a
backend: MockBackend by default, or MsSwiftBackend (``swift`` CLIs) when
``RSI_BACKEND=swift``.
"""

__version__ = '0.1.0'
