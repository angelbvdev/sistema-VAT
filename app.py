import os
import gc
import torch
import whisper
from moviepy import VideoFileClip
from tqdm import tqdm

# --- CONFIGURACIÓN ---
CARPETA_VIDEOS = "videos"
CARPETA_AUDIO = "audio"
CARPETA_TEXTO = "texto"
# Modelo: "base", "small", "medium" o "large" cambiar según capacidad de GPU 
# base: 1.5GB VRAM, small: 2.5GB, medium: 5GB, large: 10GB+
MODELO_WHISPER = "small" 

# --- INICIO ---
# Crear carpetas si no existen
os.makedirs(CARPETA_AUDIO, exist_ok=True)
os.makedirs(CARPETA_TEXTO, exist_ok=True)

# Detectar GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✨ Usando dispositivo: {DEVICE}")

# Cargar modelo (fuera del bucle para hacerlo solo una vez)
print(f"✨ Cargando modelo Whisper ({MODELO_WHISPER})...")
try:
    modelo = whisper.load_model(MODELO_WHISPER, device=DEVICE)
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    exit()

# Listar videos
extensiones_validas = (".mp4", ".mov", ".mkv", ".avi", ".flv", ".webm")
videos = [f for f in os.listdir(CARPETA_VIDEOS) if f.lower().endswith(extensiones_validas)]
print(f"✨ Se encontraron {len(videos)} video(s) en '{CARPETA_VIDEOS}'.")

# Procesar videos con barra de progreso
pbar = tqdm(videos, desc="Procesando")

for video in pbar:
    ruta_video = os.path.join(CARPETA_VIDEOS, video)
    nombre_base = os.path.splitext(video)[0]
    
    nombre_audio = nombre_base + ".wav"
    ruta_audio = os.path.join(CARPETA_AUDIO, nombre_audio)
    
    nombre_texto = nombre_base + ".txt"
    ruta_texto = os.path.join(CARPETA_TEXTO, nombre_texto)

    # 1️⃣ VERIFICACIÓN: Si el texto ya existe, saltamos
    if os.path.exists(ruta_texto):
        pbar.write(f"⏩ Saltando (ya existe): {video}")
        continue

    try:
        # Actualizar descripción de la barra
        pbar.set_description(f"📂 Extrayendo audio: {video[:15]}...")

        # 2️⃣ EXTRAER AUDIO
        # Usamos 'logger=None' para silenciar MoviePy 2.0+ y evitar el error de 'verbose'
        clip = VideoFileClip(ruta_video)
        clip.audio.write_audiofile(ruta_audio, logger=None, codec='pcm_s16le')
        clip.close() 
        del clip # Eliminar objeto para liberar memoria

        # 3️⃣ TRANSCRIBIR
        pbar.set_description(f"🧠 Transcribiendo: {video[:15]}...")
        resultado = modelo.transcribe(ruta_audio, fp16=(DEVICE == "cuda"))
        texto = resultado["text"]

        # 4️⃣ GUARDAR TEXTO
        with open(ruta_texto, "w", encoding="utf-8") as f:
            f.write(texto.strip())
        
        pbar.write(f"✅ Completado: {nombre_texto}")

    except Exception as e:
        pbar.write(f"❌ Error con {video}: {str(e)}")
        continue
        
    finally:
        # 5️⃣ LIMPIEZA DE MEMORIA (Crucial para GPU de 6GB)
        if os.path.exists(ruta_audio):
            # Opcional: Borrar el audio wav intermedio para ahorrar espacio en disco
            # os.remove(ruta_audio) 
            pass
            
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

print("\n🌟 ¡Proceso finalizado con éxito!")