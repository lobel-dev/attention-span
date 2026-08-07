#!/usr/bin/env python3
"""Executable shim for the packaged attention-span updater."""

import os
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from attention_span.update import main as package_main

    if argv is None:
        return package_main()
    return package_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
