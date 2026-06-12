"""Entry point for the standalone PyQt5 GUI."""
from __future__ import annotations

import sys

from PyQt5.QtWidgets import QApplication

from .widget import DgPidTunerWidget


def main(args=None) -> int:
    app = QApplication(sys.argv if args is None else list(args))
    w = DgPidTunerWidget()
    w.resize(1200, 760)
    w.show()
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
