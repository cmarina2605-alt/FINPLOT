#!/bin/bash
echo "FINPILOT AI - INSTALACIÓN AUTOMÁTICA"
echo "=========================================="

# 1. Crear entorno virtual
echo "Creando entorno virtual..."
python3 -m venv venv

# 2. Activar entorno
echo "Activando entorno..."
source venv/bin/activate

# 3. Actualizar pip
echo "Actualizando pip..."
pip install --upgrade pip

# 4. Instalar dependencias
echo "Instalando dependencias..."
pip install --force-reinstall -r requirements.txt

# 5. Finalizado
echo ""
echo "¡INSTALACIÓN COMPLETA!"
echo "=========================================="
echo "AHORA EJECUTA:"
echo "   ./start.sh"
echo ""
echo "Modelos disponibles:"
echo "   ollama pull gemma2:9b   # 5.4 GB"
echo "   ollama pull gemma2:2b   # 1.3 GB"
echo ""