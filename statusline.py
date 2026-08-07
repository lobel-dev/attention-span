#!/usr/bin/env python3
"""Executable shim for the packaged attention-span statusline."""

import os
import sys


def main() -> None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from attention_span.statusline import main as package_main

    package_main()


if __name__ == "__main__":
    main()
