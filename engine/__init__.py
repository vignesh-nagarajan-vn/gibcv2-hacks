"""Mirage engine.

Offline numerical core for the backtest overfitting auditor. Every module here
runs on a developer machine or in CI. Nothing in this package is imported by the
web app at runtime. The only channel between the two halves is the JSON written
into results/.
"""

__version__ = "0.1.0"
