@echo off
echo INSTALANDO FINPILOT AI...
python -m venv venv
call venv\Scripts\activate
pip install --upgrade pip
pip install --force-reinstall -r requirements.txt

echo.
echo LISTO! EJECUTA start.bat
cmd /k