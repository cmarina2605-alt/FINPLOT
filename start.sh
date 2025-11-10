#!/bin/bash
echo
echo "============================================="
echo "   FINPILOT AI - START (ALWAYS WORKS)"
echo "============================================="
echo

# 2. Activate Python virtual environment
source venv/bin/activate

# 3. Enable offline mode for Ollama (prevents mirror connection)
echo "[CONFIG] Setting Ollama to OFFLINE mode..."
export OLLAMA_NO_PULL=true
export OLLAMA_HOST="http://0.0.0.0:11434"

# 4. Start Ollama in background (if not already running)
echo "[OLLAMA] Checking if Ollama is already running..."
if lsof -i:11434 >/dev/null 2>&1; then
  echo "[INFO] Ollama is already running."
else
  echo "[START] Launching Ollama server..."
  nohup ollama serve > ollama.log 2>&1 &
fi

# 5. Wait for Ollama to be ready (max 60 sec)
echo
echo "[WAITING] Ollama is starting..."
for i in {1..20}; do
  if lsof -i:11434 >/dev/null 2>&1; then
    echo "[OK] Ollama ready! (after $i attempts)"
    break
  fi
  if [ $i -eq 20 ]; then
    echo "[ERROR] Ollama did not respond after 60 seconds."
    echo "        Run manually: ollama serve"
    exit 1
  fi
  echo ".... waiting for Ollama ($i/20)"
  sleep 3
done

# 6. Clean up port 8000 (kill previous process)
echo
echo "[CLEANUP] Releasing port 8000..."
PID=$(lsof -ti:8000)
if [ -n "$PID" ]; then
  kill -9 "$PID" >/dev/null 2>&1
fi

# 7. Open the browser automatically
echo
echo "[BROWSER] Opening http://127.0.0.1:8000"
if command -v xdg-open >/dev/null; then
  xdg-open "http://127.0.0.1:8000"
elif command -v open >/dev/null; then
  open "http://127.0.0.1:8000"
fi

# 8. Start the FinPilot API server
echo
echo "[FINPILOT] STARTING API - CHAT READY TO USE!"
echo
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
