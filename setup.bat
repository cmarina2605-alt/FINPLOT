@echo off
echo INSTALANDO FINPILOT AI...
python -m venv venv
call venv\Scripts\activate
pip install --upgrade pip
pip install --force-reinstall -r requirements.txt
pip install --force-reinstall numpy==1.26.4
echo.
echo LISTO! EJECUTA start.bat
cmd /k