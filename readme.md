# FinPilot AI – Tu Asesor Financiero con RAG + LLM Local

## Modelos disponibles
- **gemma2:9b** – 9B parámetros | 5.4 GB | ~1.2 seg por respuesta (Alta precisión)  
- **gemma2:2b** – 2B parámetros | 1.3 GB | ~0.6 seg por respuesta (Ultra rápido)  

## Características principales
- 100% local → 0 internet  
- Integración **RAG + LLM**  
- **HYDE + Cross-Encoder** para mejorar contexto  
- **Revisor CNMV** para cumplimiento regulatorio  

---

## ¿Qué hace FinPilot AI?

1. Subes un PDF financiero (ej: `RiskProfiles.pdf`).  
2. Realizas preguntas en lenguaje natural.  
3. Obtienes respuestas **con cita exacta de página** del documento.  
4. Puedes pulsar **REVISAR (CNMV)** para validar la respuesta → tick verde si cumple regulaciones.

---

## Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/cmarina2605-alt/FinPilotIA
cd FinPilotIA
```

### 2. Crear y activar entorno virtual

**Windows:**
```bat
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Instalar Ollama
Descargar e instalar Ollama desde [https://ollama.com](https://ollama.com)

### 5. Descargar modelos

**Modelo de alta precisión:**
```bash
ollama pull gemma2:9b   # 5.4 GB → 1.2 seg por respuesta
```

**Modelo rápido:**
```bash
ollama pull gemma2:2b   # 1.3 GB → 0.6 seg por respuesta
```

### 6. Ejecutar la aplicación

**Windows:**
```bat
.\setup.bat
```

**macOS / Linux:**
```bash
./setup.sh
```

### 7. Iniciar FastAPI
```bash
uvicorn app:app --reload
```

---

## Uso


1. Abrir el navegador en [http://127.0.0.1:8000](http://127.0.0.1:8000)  
2. Subir un PDF financiero en la sección **Subir PDF**.  
3. Hacer preguntas en lenguaje natural.  
4. Revisar la respuesta con **REVISAR (CNMV)**.  
5. Cambiar modelo si deseas velocidad o precisión distinta en **Cambiar Modelo**.