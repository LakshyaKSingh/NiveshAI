"""
TradePilot Training Entrypoint (Thin Wrapper)
=============================================
Delegates execution to the modular package pipeline:
`tradepilot.pipelines.train_pipeline`

Usage:
    python train.py
    python -m tradepilot.pipelines.train_pipeline
"""

from tradepilot.pipelines.train_pipeline import main

if __name__ == "__main__":
    main()
