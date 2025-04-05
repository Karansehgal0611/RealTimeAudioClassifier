import pyaudio
import numpy as np
import librosa
import tensorflow as tf
import time

# Constants
RATE = 22050
CHUNK = 1024  # Must be power of 2 (512, 1024, 2048)
BUFFER_SECONDS = 1.0
BUFFER_SIZE = int(RATE * BUFFER_SECONDS)
THRESHOLD = 0.7

# Verify buffer alignment
assert BUFFER_SIZE % CHUNK == 0, "Buffer size must be divisible by chunk size"
print(f"Buffer: {BUFFER_SIZE} samples ({BUFFER_SIZE/RATE:.2f}s)")
print(f"Chunks per buffer: {BUFFER_SIZE//CHUNK}")

# Initialize model
model = tf.keras.models.load_model('emergency_sound_classifier.h5')

# Audio setup
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# Circular buffer implementation
audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
write_ptr = 0
chunk_counter = 0

def process_audio(buffer):
    # Create spectrogram
    spect = librosa.feature.melspectrogram(y=buffer, sr=RATE, n_mels=64)
    spect = librosa.power_to_db(spect, ref=np.max)
    spect = cv2.resize(spect, (64, 64))
    
    # Safe normalization
    spect = np.nan_to_num(spect)
    if np.ptp(spect) > 0:
        spect = (spect - spect.min()) / np.ptp(spect)
    else:
        spect = np.zeros((64, 64))
    return np.expand_dims(spect, axis=-1)

print("🎙️ Listening... (Press Ctrl+C to stop)")

try:
    while True:
        try:
            # Read audio chunk
            data = stream.read(CHUNK, exception_on_overflow=False)
            chunk = np.frombuffer(data, dtype=np.float32)
            
            # Write to circular buffer
            if write_ptr + CHUNK <= BUFFER_SIZE:
                audio_buffer[write_ptr:write_ptr+CHUNK] = chunk
            else:
                remaining = BUFFER_SIZE - write_ptr
                audio_buffer[write_ptr:] = chunk[:remaining]
                audio_buffer[:CHUNK-remaining] = chunk[remaining:]
            
            write_ptr = (write_ptr + CHUNK) % BUFFER_SIZE
            chunk_counter += 1
            
            print(f"{RATE} samples/sec ÷ {CHUNK} samples/chunk = {RATE/CHUNK:.2f} chunks/sec")
            print(f"Exact buffer: {BUFFER_SIZE} samples = {BUFFER_SIZE/CHUNK} chunks")
            # Process every full buffer cycle
            if chunk_counter >= BUFFER_SIZE // CHUNK:
                spect = process_audio(audio_buffer)
                prob = model.predict(spect[np.newaxis, ...], verbose=0)[0][0]
                
                if prob > THRESHOLD:
                    print(f"🚨 EMERGENCY ({prob:.0%} confidence)")
                else:
                    print(f"Normal ({prob:.0%})", end='\r')
                
                chunk_counter = 0
                
        except Exception as e:
            print(f"\nError: {str(e)}")
            time.sleep(0.1)
            # Reset buffer on serious errors
            audio_buffer.fill(0)
            write_ptr = 0
            chunk_counter = 0

except KeyboardInterrupt:
    print("\nStopping...")
    stream.stop_stream()
    stream.close()
    p.terminate()