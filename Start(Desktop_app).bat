@echo off
REM === Imposta il percorso specifico di Python ===
set PYTHON_EXE="C:\Users\francesco.vallisa\AppData\Local\Programs\Python\Python311\python.exe"

REM === Vai nella cartella dello script ===
cd "C:\Users\francesco.vallisa\OneDrive - Angel Company\Desktop\s.w\analizzatore_csv_v1"

REM === Esegui lo script con quella versione di Python ===
%PYTHON_EXE% main.py

pause
