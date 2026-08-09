# -*- coding: utf-8 -*-
"""
run.py
------
UTF-8 safe launcher for the AlphaSignal Engine dashboard.
Run this instead of calling streamlit directly to avoid
Windows cp1252 UnicodeEncodeError.

Usage:
    python run.py
"""
import os
import sys
import subprocess

# Force UTF-8 for stdout/stderr before spawning streamlit
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

subprocess.run(
    [sys.executable, "-X", "utf8", "-m", "streamlit", "run",
     "dashboard/app.py", "--server.port", "8501", "--server.headless", "true"],
    env=os.environ,
)
