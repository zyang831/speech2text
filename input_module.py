import pyaudio
import numpy as np
import torch
import librosa
import transformers
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

class AudioPreprocessor:
    def __init__(self, 
                input_rate=44100, 
                target_rate=16000, 
                model_name="facebook/wav2vec2-base-960h"):
        """
        Initialize audio preprocessor for real-time streaming
        
        Args:
            input_rate (int): Input audio sampling rate
            target_rate (int): Target sampling rate for model
            model_name (str): Hugging Face model identifier
        """
        self.input_rate = input_rate
        self.target_rate = target_rate
        
        # Resampling parameters
        self.resample_ratio = target_rate / input_rate
        
        # Noise reduction parameters
        self.noise_threshold = 0.02
        
        # Streaming buffer management
        self.audio_buffer = []
        self.buffer_duration_sec = 1.0  # 1-second sliding window
        self.buffer_max_length = int(self.buffer_duration_sec * target_rate)
        
        # Load transformer preprocessing components
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
    
    def preprocess_chunk(self, audio_chunk):
        """
        Preprocess an incoming audio chunk
        
        Args:
            audio_chunk (bytes): Raw audio data from PyAudio
        
        Returns:
            np.ndarray: Preprocessed audio chunk
        """
        # Convert byte data to numpy array
        audio_float = np.frombuffer(audio_chunk, dtype=np.float32)
        
        # Resample audio to target rate
        audio_resampled = librosa.resample(
            audio_float, 
            orig_sr=self.input_rate, 
            target_sr=self.target_rate
        )
        
        # Noise reduction
        audio_denoised = self._reduce_noise(audio_resampled)
        
        return audio_denoised
    
    def _reduce_noise(self, audio):
        """
        Apply basic noise reduction
        
        Args:
            audio (np.ndarray): Input audio signal
        
        Returns:
            np.ndarray: Noise-reduced audio
        """
        # Remove low-amplitude noise
        return np.where(
            np.abs(audio) > self.noise_threshold, 
            audio, 
            0
        )
    
    def update_buffer(self, preprocessed_chunk):
        """
        Manage sliding window buffer for continuous processing
        
        Args:
            preprocessed_chunk (np.ndarray): Preprocessed audio chunk
        
        Returns:
            np.ndarray or None: Full buffer ready for model input
        """
        # Append new chunk to buffer
        self.audio_buffer.extend(preprocessed_chunk)
        
        # Trim buffer to max length
        if len(self.audio_buffer) > self.buffer_max_length:
            self.audio_buffer = self.audio_buffer[-self.buffer_max_length:]
        
        # Check if buffer is full enough for processing
        if len(self.audio_buffer) >= self.buffer_max_length:
            return np.array(self.audio_buffer)
        
        return None
    
    def prepare_model_input(self, audio_buffer):
        """
        Prepare audio buffer for transformer model input
        
        Args:
            audio_buffer (np.ndarray): Preprocessed audio buffer
        
        Returns:
            dict: Transformer model input
        """
        # Normalize audio
        audio_normalized = audio_buffer / np.max(np.abs(audio_buffer))
        
        # Convert to model input
        model_input = self.processor(
            audio_normalized, 
            sampling_rate=self.target_rate, 
            return_tensors="pt"
        )
        
        return model_input
    
    def predict(self, model_input):
        """
        Run inference on preprocessed audio
        
        Args:
            model_input (dict): Preprocessed audio input
        
        Returns:
            str: Predicted transcription
        """
        # Run model inference
        with torch.no_grad():
            logits = self.model(model_input.input_values).logits
        
        # Decode prediction
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        
        return transcription

def record_audio():
    """
    Main audio recording function with real-time preprocessing
    """
    # Audio parameters
    CHUNK = 1024  # Slightly adjusted for better processing
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 44100

    # Initialize preprocessor
    preprocessor = AudioPreprocessor(input_rate=RATE)

    p = pyaudio.PyAudio()

    try:
        # Open input stream
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )

        print("* Recording audio... Press Ctrl+C to stop")

        while True:
            # Read audio data
            data = stream.read(CHUNK)
            
            # Preprocess chunk
            preprocessed_chunk = preprocessor.preprocess_chunk(data)
            
            # Update sliding window buffer
            buffer = preprocessor.update_buffer(preprocessed_chunk)
            
            # Process if buffer is ready
            if buffer is not None:
                try:
                    # Prepare for model input
                    model_input = preprocessor.prepare_model_input(buffer)
                    
                    # Run inference
                    transcription = preprocessor.predict(model_input)
                    
                    # Print transcription (or handle as needed)
                    # print(f"Transcription: {transcription}")
                
                except Exception as inference_error:
                    print(f"Inference error: {inference_error}")

    except KeyboardInterrupt:
        print("\n* Recording stopped")
    except Exception as e:
        print(f"Error recording: {e}")
    finally:
        # Cleanup
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    record_audio()