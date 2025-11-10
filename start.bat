@echo off
echo.
echo =============================================
echo    FINPILOT AI - START (ALWAYS WORKS)
echo =============================================
echo.

:: 1. Go to the correct working directory
cd /d "C:\temp\FINPILOT\FINPILOT"

:: 2. Activate the Python virtual environment
call venv\Scripts\activate

:: 3. Enable offline mode for Ollama (prevents mirror connection)
echo [CONFIG] Setting Ollama to OFFLINE mode...
setx OLLAMA_NO_PULL true >nul


:: 3a. Establecer puerto
set "OLLAMA_HOST=http://0.0.0.0:11434"

:: 4. Start Ollama using PowerShell (ALWAYS opens a window)
echo [OLLAMA] Opening OLLAMA SERVER window...
powershell -Command "Start-Process cmd -ArgumentList '/c title OLLAMA SERVER && color 0a && echo OLLAMA RUNNING (OFFLINE MODE)... && ollama serve' -WindowStyle Normal"

:: 5. Wait for Ollama to be ready (up to 60 seconds)
echo.
echo [WAITING] Ollama is starting (check the green window)...
set "count=0"
:wait
set /a count+=1
timeout /t 3 >nul
netstat -ano | findstr :11434 | findstr LISTENING >nul
if %errorlevel% equ 0 (
    echo.
    echo [OK] Ollama ready! (after %count% attempts)
    goto continue
)
if %count% geq 20 (
    echo.
    echo [ERROR] Ollama did not respond after 60 seconds.
    echo         Open a terminal manually and run: ollama serve
    pause
    exit /b 1
)
echo .... waiting for Ollama (%count%/20)...
goto wait

:continue
:: 6. Clean up port 8000 (kill previous processes)
echo.
echo [CLEANUP] Releasing port 8000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do taskkill /F /PID %%a >nul 2>&1

:: 7. Open the browser automatically
echo.
echo [BROWSER] Opening http://127.0.0.1:8000
start http://127.0.0.1:8000

:: 8. Start the FinPilot API server
echo.
echo [FINPILOT] STARTING API - CHAT READY TO USE!
echo.
uvicorn app:app --host 127.0.0.1 --port 8000 --reload

pause
