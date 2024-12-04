# Python
import numpy as np
import torch
import sounddevice as sd
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from collections import deque
import threading
import queue
import librosa
import noisereduce as nr

class RealTimeASR:
    def __init__(self, 
                 model_name='facebook/wav2vec2-large-960h-lv60-self', 
                 sample_rate=16000, 
                 chunk_duration=1.0,
                 max_buffer_size=16000 * 5,
                 callback=None):  # Add callback parameter

        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration)
        self.max_buffer_size = max_buffer_size

        self.running_buffer = deque(maxlen=self.max_buffer_size)
        self.audio_queue = queue.Queue()
        self.is_running = True
        self.callback = callback  # Store callback function
        self.noise_sample = None
        self.calibration_samples = []
        self.is_calibrating = True
        self.calibration_duration = 1  # 1 second of noise sampling

    def preprocess_audio(self, audio_data):
        # Normalize audio
        audio_data = librosa.util.normalize(audio_data)
        
        # Apply noise reduction if we have a noise profile
        if self.noise_sample is not None:
            audio_data = nr.reduce_noise(
                y=audio_data,
                sr=self.sample_rate,
                prop_decrease=0.8
            )
        
        return audio_data

    def process_audio_stream(self, indata, frames, time, status):
        if status:
            print(f"Audio Stream Error: {status}")
            return

        audio_chunk = indata.flatten()

        # During calibration, collect noise samples
        if self.is_calibrating:
            self.calibration_samples.extend(audio_chunk)
            if len(self.calibration_samples) >= self.sample_rate * self.calibration_duration:
                print("Calibration complete. Beginning transcription...")
                self.noise_sample = np.array(self.calibration_samples)
                self.is_calibrating = False
            return

        # Preprocess audio before adding to queue
        processed_audio = self.preprocess_audio(audio_chunk)
        self.audio_queue.put(processed_audio)

    def process_queue(self):
        while self.is_running:
            try:
                audio_data = self.audio_queue.get(timeout=1.0)
                self.running_buffer.extend(audio_data)

                if len(self.running_buffer) >= self.chunk_size:
                    current_chunk = np.array(self.running_buffer)
                    self.running_buffer.clear()

                    inputs = self.processor(
                        current_chunk, 
                        sampling_rate=self.sample_rate, 
                        return_tensors="pt", 
                        padding=True  # Ensure proper padding
                    )
                    with torch.no_grad():
                        logits = self.model(inputs.input_values).logits
                    predicted_ids = torch.argmax(logits, dim=-1)
                    transcription = self.processor.decode(predicted_ids[0])
                    
                    # Call the callback function if it exists
                    if self.callback:
                        self.callback(transcription)
                    else:
                        print(f"Transcription: {transcription}")
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing audio: {e}")
        
    def start_streaming(self):
        print("Calibrating noise profile... Please remain quiet.")
        process_thread = threading.Thread(target=self.process_queue)
        process_thread.start()
        
        try:
            with sd.InputStream(
                callback=self.process_audio_stream,
                channels=1,
                samplerate=self.sample_rate
            ):
                print("Streaming started. Press Ctrl+C to stop.")
                while self.is_running:
                    sd.sleep(100)
        except KeyboardInterrupt:
            print("\nStopping stream...")
        finally:
            self.is_running = False
            process_thread.join()

if __name__ == "__main__":
    real_time_asr = RealTimeASR()
    real_time_asr.start_streaming()