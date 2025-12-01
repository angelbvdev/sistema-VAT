🎥 V-A-T: Video to Audio to Text Transcriber

Un herramienta robusta en Python diseñada para automatizar la extracción de audio de archivos de video y transcribirlos a texto usando el modelo Whisper de OpenAI.

Optimizado para GPU NVIDIA (CUDA) y las últimas versiones de MoviePy, con manejo inteligente de memoria y compatibilidad para VRAM limitada (4–6 GB).

🚀 Características

Procesamiento por lotes: Escanea automáticamente la carpeta videos/ para archivos .mp4, .mkv, .mov, etc.

Aceleración GPU: Compatible con NVIDIA CUDA usando PyTorch.

Reanudación inteligente: Omite archivos ya transcritos para ahorrar tiempo y recursos.

Gestión de memoria: Previene saturación de VRAM, ideal para GPUs con 4–6 GB.

MoviePy 2.x compatible: Usando la importación moderna y funciones actualizadas.

📂 Estructura de carpetas

videos/ – Carpeta para tus videos de entrada. Se sube vacía con .gitkeep.

audio/ – Carpeta para los audios extraídos (.wav). Se sube vacía con .gitkeep.

texto/ – Carpeta para las transcripciones (.txt). Se sube vacía con .gitkeep.

app.py – Script principal que procesa videos, extrae audio y genera texto.

requirements.txt – Dependencias del proyecto.

.gitignore – Ignora los archivos generados dentro de videos/, audio/ y texto/, pero mantiene las carpetas.

🛠️ Tecnologías

Python 3.10+

OpenAI Whisper – Sistema de reconocimiento automático de voz (ASR) de última generación.

MoviePy 2.x – Extracción eficiente de audio de videos.

PyTorch – Computación de tensores y aceleración GPU.

TQDM – Visualización de progreso.

📋 Requisitos previos

Instala FFmpeg:

Ubuntu/Debian:

sudo apt update
sudo apt install ffmpeg


Windows:
Descarga FFmpeg desde ffmpeg.org
 y añádelo al PATH del sistema.

⚙️ Instalación

Clona el repositorio:

git clone https://github.com/angelbvdev/sistema-VAT.git
cd sistema-VAT


Crea un entorno virtual (opcional pero recomendado):

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate


Instala las dependencias:

pip install -r requirements.txt

▶️ Uso

Coloca tus videos en videos/.

Ejecuta el script:

python app.py


Se generarán automáticamente dos carpetas:

audio/: Contiene los archivos .wav extraídos.

texto/: Contiene las transcripciones .txt.

🔧 Configuración

Puedes ajustar el tamaño del modelo Whisper en app.py según tu GPU:

# Opciones: "tiny", "base", "small", "medium", "large"
modelo = whisper.load_model("small", device=DEVICE)

🤝 Contribuciones

¡Bienvenidas! Envía un Pull Request para mejoras o nuevas funcionalidades.

📄 Licencia

Este proyecto está bajo MIT License.