# CaliGuia F5-TTS Worker

Worker FastAPI para síntesis de voz en CaliGuia usando F5-TTS y el checkpoint en español `jpgallegoar/F5-Spanish`.

La superapp envía una sola grabación de referencia con su transcripción exacta, y luego reutiliza esa misma voz para sintetizar nuevos diálogos generados por IA. F5-TTS usa el audio de referencia para identidad/prosodia, `reference_text` para alinear texto y audio, y `text` como el contenido nuevo que debe decir.

## Qué Hace Este Servicio

- Recibe desde `caliguia-superapp` el texto que se quiere sintetizar.
- Recibe el audio de referencia de la voz seleccionada.
- Normaliza el audio de referencia a WAV mono de 24 kHz con FFmpeg.
- Valida muestras de voz del usuario transcribiéndolas con Whisper y comparándolas con el texto esperado.
- Limpia texto generado por IA antes de enviarlo a F5-TTS.
- Divide diálogos largos en frases cortas para mejorar alineación y fluidez.
- Ejecuta F5-TTS con pesos de F5-Spanish y devuelve `audio/wav`.

## Endpoints

### `GET /health`

Devuelve el estado básico del servicio:

```json
{
  "ok": true,
  "engine": "f5-tts",
  "model": "F5TTS_Base"
}
```

### `POST /validate-reference`

Valida que el usuario haya leído correctamente el texto de referencia.

Campos multipart:

- `speaker_wav`: grabación del usuario.
- `reference_text`: texto exacto que el usuario debía leer.
- `language`: `es`, `en` o `pt`; por defecto `es`. Controla el idioma usado por Whisper en la validación.

Respuesta:

```json
{
  "accepted": true,
  "match_score": 0.94,
  "threshold": 0.82,
  "transcription": "..."
}
```

La superapp llama este endpoint antes de guardar un perfil de voz. Así evitamos reutilizar una muestra mal leída en todas las narraciones futuras. Para usuarios multilingües, envía el mismo idioma usado por el texto de referencia para que Whisper valide español, inglés o portugués con el prompt correcto.

### `POST /tts`

Sintetiza voz.

Campos multipart:

- `text`: texto nuevo que debe decir la voz.
- `language`: `es`, `en` o `pt`; por defecto `es`.
- `voice_id`: id opcional de voz, por ejemplo `system:jorge` o `xtts-local:{userId}:{voiceKey}`.
- `speaker_wav`: audio de referencia.
- `reference_text`: transcripción exacta de `speaker_wav`.
- `ref_text` / `speaker_text`: alias de compatibilidad para `reference_text`.

Devuelve:

- `Content-Type: audio/wav`
- archivo: `caliguia-voice.wav`

## Configuración en la Superapp

En `caliguia-superapp`, apunta la app a este worker:

```env
F5_TTS_API_URL=http://127.0.0.1:8010/tts
XTTS_TIMEOUT_MS=120000
```

Para Cloud Run u otro worker desplegado:

```env
F5_TTS_API_URL=https://YOUR-VOICE-WORKER.run.app/tts
XTTS_TIMEOUT_MS=300000
```

## Variables de Entorno

Los valores por defecto ya están configurados en ambos Dockerfiles.

```env
F5_TTS_MODEL_NAME=F5TTS_Base
F5_TTS_CKPT_FILE=/app/models/F5-Spanish/model_1200000.safetensors
F5_TTS_VOCAB_FILE=/app/models/F5-Spanish/vocab.txt
F5_TTS_MAX_TEXT_CHARS=900
F5_TTS_MAX_CHUNK_CHARS=220
F5_TTS_MIN_REFERENCE_MATCH_SCORE=0.82
F5_TTS_NFE_STEP=64
F5_TTS_CFG_STRENGTH=1.5
F5_TTS_SPEED=0.92
F5_TTS_SWAY_SAMPLING_COEF=-1.0
F5_TTS_REMOVE_SILENCE=1
F5_TTS_CROSS_FADE_DURATION=0.15
F5_TTS_PRELOAD_ON_STARTUP=1
```

Notas de calidad:

- `F5_TTS_NFE_STEP=64` prioriza calidad y estabilidad sobre latencia.
- `F5_TTS_SWAY_SAMPLING_COEF=-1.0` aplica Sway Sampling para reforzar la alineación temprana.
- `F5_TTS_MAX_CHUNK_CHARS=220` mantiene diálogos en fragmentos cortos y más fieles.
- `F5_TTS_MIN_REFERENCE_MATCH_SCORE=0.82` define qué tan estricta es la validación de lectura.

## Ejecutar Local Sin Docker

Desde esta carpeta del worker:

```powershell
python -m venv tts
.\tts\Scripts\python -m pip install --upgrade pip
.\tts\Scripts\python -m pip install -r requirements.txt
.\tts\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8010
```

Verifica el servicio:

```powershell
curl http://127.0.0.1:8010/health
```

## Ejecutar Con `Dockerfile.local` GPU

`Dockerfile.local` es la imagen CUDA/GPU. Úsala para desarrollo local con GPU NVIDIA y mejor latencia.

Construir:

```powershell
docker build -f Dockerfile.local -t caliguia-tts-worker:gpu .
```

Ejecutar:

```powershell
docker run --rm --gpus all -p 8010:8080 --name caliguia-tts-gpu caliguia-tts-worker:gpu
```

Verificar:

```powershell
curl http://127.0.0.1:8010/health
```

Notas:

- Requiere Docker con NVIDIA Container Toolkit.
- La imagen descarga `model_1200000.safetensors` y `vocab.txt` desde `jpgallegoar/F5-Spanish` durante el build.
- El modelo se valida durante el build en CPU, pero corre en CUDA al iniciar porque `F5_TTS_FORCE_CPU=0`.

## Ejecutar Con `Dockerfile` CPU

`Dockerfile` es la imagen CPU basada en `python:3.11-slim`. Úsala cuando no haya GPU, para pruebas de compatibilidad o despliegues de bajo tráfico.

Construir:

```powershell
docker build -f Dockerfile -t caliguia-tts-worker:cpu .
```

Ejecutar:

```powershell
docker run --rm -p 8010:8080 --name caliguia-tts-cpu caliguia-tts-worker:cpu
```

Verificar:

```powershell
curl http://127.0.0.1:8010/health
```

Notas:

- La síntesis en CPU es mucho más lenta que en GPU.
- La imagen instala PyTorch CPU y usa `F5_TTS_FORCE_CPU=1`.
- Si usas CPU, aumenta `XTTS_TIMEOUT_MS` en la superapp.

## Ejemplo de Request a `/tts`

```powershell
curl -X POST http://127.0.0.1:8010/tts `
  -F "text=Estamos cerca del Bulevar del Rio. Mira hacia la izquierda y camina con calma." `
  -F "language=es" `
  -F "reference_text=Hola, soy la voz de CaliGuia. Caminare contigo por Cali con calma, curiosidad y mucho sabor local." `
  -F "speaker_wav=@C:\tmp\reference.wav" `
  --output C:\tmp\caliguia-voice.wav
```

## Cloud Run

Despliega esta carpeta como un servicio independiente de Cloud Run y apunta la superapp al endpoint `/tts`.

Recursos iniciales recomendados:

```txt
CPU: 2+
Memoria: 4Gi mínimo, 8Gi recomendado
Concurrencia: 1 para CPU, 1-4 para GPU
Timeout: 300s
```

Para calidad de producción, usa GPU. CPU funciona, pero validación y síntesis pueden superar timeouts cortos.

## Checklist de Calidad de Voz

- Usa un único texto fijo de referencia en la superapp.
- Mantén la grabación limpia, cerca del micrófono, sin música, sin eco y sin silencios largos.
- Valida la grabación contra `reference_text` antes de guardarla.
- Mantén los diálogos generados por IA en frases claras y concisas.
- Evita markdown, URLs, emojis y abreviaturas crudas en el texto enviado a TTS.
- Mantén emparejados el checkpoint y el vocabulario de F5-Spanish.
