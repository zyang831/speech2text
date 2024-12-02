import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, wiener
from datasets import load_dataset
import io
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio

# Load the audio dataset
def load_audio_dataset():
    try:
        dataset = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k")
        train_dataset = dataset['train']
        if len(train_dataset) == 0:
            raise ValueError("Dataset is empty")
        return train_dataset
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise

# Load audio data from dataset sample
def load_audio_from_dataset(sample, audio_type='noisy'):
    """Extract audio data and sampling rate from a dataset sample"""
    try:
        audio_array = sample[audio_type]['array']
        sampling_rate = sample[audio_type]['sampling_rate']
        return audio_array, sampling_rate
    except Exception as e:
        print(f"Error loading audio data: {str(e)}")
        return None, None

class AudioDataset(Dataset):
    """
    A custom dataset class for loading and processing audio data.
    Args:
        dataset (list): A list of dataset samples.
        transform (callable, optional): A function/transform to apply to the audio data.
        fixed_length (int, optional): The fixed length to pad or trim the audio signals to. Default is 66560.
    Methods:
        __len__(): Returns the total number of samples in the dataset.
        __getitem__(idx): Retrieves the noisy and clean audio samples at the specified index.
    Attributes:
        dataset (list): The dataset containing audio samples.
        transform (callable): The transform function to apply to the audio data.
        fixed_length (int): The fixed length to pad or trim the audio signals to.
    """
    def __init__(self, dataset, transform=None, fixed_length=66560):
        self.dataset = dataset
        self.transform = transform
        self.fixed_length = fixed_length - (fixed_length % 2048)  # Ensure fixed_length is a multiple of 2048
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        noisy_audio, sr = load_audio_from_dataset(sample, 'noisy')
        clean_audio, _ = load_audio_from_dataset(sample, 'clean')
        
        # Convert to torch tensors
        noisy_audio = torch.FloatTensor(noisy_audio)
        clean_audio = torch.FloatTensor(clean_audio)
        
        # Pad or trim to fixed length
        noisy_audio = self._pad_or_trim(noisy_audio, self.fixed_length)
        clean_audio = self._pad_or_trim(clean_audio, self.fixed_length)
        
        # Reshape to match the input size of the autoencoder
        noisy_audio = noisy_audio.view(1, -1)  # Add channel dimension
        clean_audio = clean_audio.view(1, -1)  # Add channel dimension
        
        if self.transform:
            noisy_audio = self.transform(noisy_audio)
            clean_audio = self.transform(clean_audio)
            
        return noisy_audio, clean_audio
    
    def _pad_or_trim(self, audio, length):
        if len(audio) > length:
            return audio[:length]
        elif len(audio) < length:
            return F.pad(audio, (0, length - len(audio)))
        else:
            return audio

class SpectralSubtraction:
    """
    A class to perform spectral subtraction for speech enhancement.
    Attributes:
    -----------
    n_fft : int
        Number of FFT points. Default is 2048.
    hop_length : int
        Number of audio samples between adjacent STFT columns. Default is 512.
    Methods:
    --------
    enhance(noisy_signal, noise_estimate=None):
        Enhances the noisy signal by performing spectral subtraction.
        Parameters:
        -----------
        noisy_signal : torch.Tensor
            The noisy input signal.
        noise_estimate : torch.Tensor, optional
            An estimate of the noise signal. If not provided, the noise is estimated from the first few frames of the noisy signal.
        Returns:
        --------
        torch.Tensor
            The enhanced signal after spectral subtraction.
    """
    def __init__(self, n_fft=2048, hop_length=512):
        self.n_fft = n_fft
        self.hop_length = hop_length
        
    def enhance(self, noisy_signal, noise_estimate=None):
        # Compute STFT
        stft = torch.stft(noisy_signal, 
                        n_fft=self.n_fft,
                        hop_length=self.hop_length,
                        return_complex=True)
        
        # Estimate noise if not provided
        if noise_estimate is None:
            # Use first few frames for noise estimation
            noise_mag = torch.abs(stft[:, :, :10]).mean(dim=-1, keepdim=True)
        else:
            noise_mag = torch.abs(torch.stft(noise_estimate, 
                                        n_fft=self.n_fft,
                                        hop_length=self.hop_length,
                                        return_complex=True))
        
        # Perform spectral subtraction
        signal_mag = torch.abs(stft)
        enhanced_mag = torch.maximum(signal_mag - noise_mag, torch.zeros_like(signal_mag))
        
        # Reconstruct signal
        enhanced_stft = enhanced_mag * (stft / (signal_mag + 1e-8))
        enhanced_signal = torch.istft(enhanced_stft, 
                                    n_fft=self.n_fft,
                                    hop_length=self.hop_length)
        
        return enhanced_signal

class DenoisingAutoencoder(nn.Module):
    """
    A denoising autoencoder neural network for reducing noise in input data.
    Args:
        input_size (int): The size of the input layer. Default is 2048.
    Attributes:
        encoder (nn.Sequential): The encoder part of the autoencoder, which compresses the input data.
        decoder (nn.Sequential): The decoder part of the autoencoder, which reconstructs the input data from the compressed representation.
    Methods:
        forward(x):
            Performs a forward pass through the autoencoder.
            Args:
                x (torch.Tensor): The input data.
            Returns:
                torch.Tensor: The reconstructed data.
    """
    def __init__(self, input_size=65536):  # Changed from 66536 to 65536
        super(DenoisingAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_size),
            nn.Tanh()
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten the input while preserving batch dimension
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(batch_size, 1, -1)  # Reshape back to (batch_size, channels, length)

class Generator(nn.Module):
    """
    A U-Net-like generator model for 1D signal processing tasks.
    Args:
        input_channels (int): Number of input channels. Default is 1.
        output_channels (int): Number of output channels. Default is 1.
    Methods:
        forward(x):
            Forward pass through the network.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, input_channels, signal_length).
            Returns:
                torch.Tensor: Output tensor of shape (batch_size, output_channels, signal_length).
    Internal Methods:
        _encoder_block(in_channels, out_channels):
            Creates an encoder block with Conv1d, LeakyReLU, and BatchNorm1d layers.
            Args:
                in_channels (int): Number of input channels.
                out_channels (int): Number of output channels.
            Returns:
                nn.Sequential: Encoder block.
        _decoder_block(in_channels, out_channels):
            Creates a decoder block with ConvTranspose1d, ReLU, and BatchNorm1d layers.
            Args:
                in_channels (int): Number of input channels.
                out_channels (int): Number of output channels.
            Returns:
                nn.Sequential: Decoder block.
    """
    def __init__(self, input_channels=1, output_channels=1):
        super(Generator, self).__init__()
        
        # Define number of features for each layer
        nf = 64  # Number of filters in first layer
        
        # Encoder blocks
        self.encoder = nn.ModuleList([
            self._encoder_block(input_channels, nf),      # nf
            self._encoder_block(nf, nf * 2),             # nf*2 
            self._encoder_block(nf * 2, nf * 4),         # nf*4
        ])
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(nf * 4, nf * 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(nf * 8),
            nn.LeakyReLU(0.2)
        )
        
        # Decoder blocks with correct input channels accounting for skip connections
        self.decoder = nn.ModuleList([
            self._decoder_block(nf * 8, nf * 4, output_padding=1),     # nf*4
            self._decoder_block(nf * 8, nf * 2, output_padding=1),     # nf*2
            self._decoder_block(nf * 4, output_channels, output_padding=0),  # Output layer
        ])
        
    def _encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def _decoder_block(self, in_channels, out_channels, output_padding=1):
        return nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, output_padding=output_padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        # Store encoder outputs for skip connections
        encoder_outputs = []
        
        # Encoder path
        for enc in self.encoder:
            x = enc(x)
            encoder_outputs.append(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path with skip connections
        for i, dec in enumerate(self.decoder):
            if i < len(self.decoder) - 1:  # Skip connection for all but last decoder layer
                # Get corresponding encoder output
                skip = encoder_outputs[-(i+1)]
                # Apply decoder
                x = dec(x)
                # Before concatenation, adjust x if necessary
                if x.size(2) != skip.size(2):
                    # Resize x to match skip
                    x = F.interpolate(x, size=skip.size(2), mode='linear', align_corners=False)
                # Concatenate skip connection
                x = torch.cat([x, skip], dim=1)
            else:
                # Last layer without skip connection
                x = dec(x)
        
        return torch.tanh(x)

class Discriminator(nn.Module):
    """
    A PyTorch implementation of a 1D Convolutional Neural Network (CNN) based Discriminator.
    Args:
        input_channels (int): Number of input channels. Default is 1.
    Attributes:
        model (nn.Sequential): The sequential container of the layers in the discriminator.
    Methods:
        forward(x):
            Defines the computation performed at every call.
            Args:
                x (torch.Tensor): Input tensor.
            Returns:
                torch.Tensor: Output tensor after passing through the model.
    """
    def __init__(self, input_channels=1):
        super(Discriminator, self).__init__()
        
        # Reduce kernel size and add padding
        self.model = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(512, 1, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),  # Flatten the output
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

class SpeechEnhancementSystem:
    """
    A system for enhancing speech signals using various techniques including spectral subtraction,
    autoencoder, and GAN-based enhancement.
    Attributes:
        device (str): The device to run the computations on ('cuda' or 'cpu').
        spectral_subtraction (SpectralSubtraction): An instance of the SpectralSubtraction class.
        autoencoder (DenoisingAutoencoder): An instance of the DenoisingAutoencoder class.
        generator (Generator): An instance of the Generator class.
        discriminator (Discriminator): An instance of the Discriminator class.
    Methods:
        train_autoencoder(train_loader, num_epochs=10, learning_rate=0.001):
            Trains the autoencoder model using the provided training data loader.
        train_gan(train_loader, num_epochs=10, learning_rate=0.0002):
            Trains the GAN model using the provided training data loader.
        enhance_speech(noisy_signal):
            Enhances the input noisy speech signal using spectral subtraction, autoencoder, and GAN-based techniques.
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.spectral_subtraction = SpectralSubtraction()
        self.autoencoder = DenoisingAutoencoder().to(device)
        self.generator = Generator().to(device)
        self.discriminator = Discriminator().to(device)
        
    def train_autoencoder(self, train_loader, num_epochs=1, learning_rate=0.001):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            for batch_idx, (noisy, clean) in enumerate(train_loader):
                noisy, clean = noisy.to(self.device), clean.to(self.device)
                
                # Forward pass
                enhanced = self.autoencoder(noisy)
                loss = criterion(enhanced, clean)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if batch_idx % 100 == 0:
                    print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
    
    def train_gan(self, train_loader, num_epochs=1, learning_rate=0.0002):
        criterion_gan = nn.BCELoss()
        criterion_l1 = nn.L1Loss()
        
        g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=learning_rate)
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            for batch_idx, (noisy, clean) in enumerate(train_loader):
                noisy, clean = noisy.to(self.device), clean.to(self.device)
                batch_size = noisy.size(0)
                
                # Ensure inputs are 3D: (batch_size, channels, time)
                noisy = noisy.view(batch_size, 1, -1)
                clean = clean.view(batch_size, 1, -1)
                
                # Train Discriminator
                d_optimizer.zero_grad()
                
                label_real = torch.ones(batch_size, 1).to(self.device)
                label_fake = torch.zeros(batch_size, 1).to(self.device)
                
                output_real = self.discriminator(clean)
                d_loss_real = criterion_gan(output_real, label_real)
                
                fake = self.generator(noisy)
                output_fake = self.discriminator(fake.detach())
                d_loss_fake = criterion_gan(output_fake, label_fake)
                
                d_loss = (d_loss_real + d_loss_fake) * 0.5
                d_loss.backward()
                d_optimizer.step()
                
                # Train Generator
                g_optimizer.zero_grad()
                
                output_fake = self.discriminator(fake)
                g_loss_gan = criterion_gan(output_fake, label_real)
                g_loss_l1 = criterion_l1(fake, clean)
                
                g_loss = g_loss_gan + 100 * g_loss_l1
                g_loss.backward()
                g_optimizer.step()
                
                if batch_idx % 100 == 0:
                    print(f'Epoch: {epoch}, Batch: {batch_idx}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')
    
    def enhance_speech(self, noisy_signal):
        """
        Enhance speech using multiple techniques
        """
        # 1. Spectral Subtraction
        enhanced_spectral = self.spectral_subtraction.enhance(noisy_signal)
        
        # 2. Autoencoder Enhancement
        with torch.no_grad():
            enhanced_auto = self.autoencoder(enhanced_spectral.to(self.device))
        
        # 3. GAN Enhancement
        with torch.no_grad():
            enhanced_gan = self.generator(enhanced_auto.unsqueeze(1))
            enhanced_gan = enhanced_gan.squeeze(1)
        
        return enhanced_gan.cpu()

def main():
    # Load dataset
    dataset = load_audio_dataset()
    audio_dataset = AudioDataset(dataset)
    train_loader = DataLoader(audio_dataset, batch_size=32, shuffle=True)
    
    # Initialize system
    system = SpeechEnhancementSystem()
    
    # Train models
    system.train_autoencoder(train_loader)
    system.train_gan(train_loader)
    
    # Example enhancement
    noisy_sample, _ = load_audio_from_dataset(dataset[0], 'noisy')
    enhanced_speech = system.enhance_speech(torch.FloatTensor(noisy_sample))
    
    return enhanced_speech

if __name__ == "__main__":
    main()
