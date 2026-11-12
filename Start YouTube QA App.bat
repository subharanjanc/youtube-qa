@echo off
title YouTube QA App
cd /d "%~dp0"
rem Uses this PC's embeddable Python when present, otherwise the "python" on PATH.
set "PY=python"
if exist "C:\Users\Windows\python\python.exe" set "PY=C:\Users\Windows\python\python.exe"
"%PY%" run_app.py
pause
