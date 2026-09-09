import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np

def load_audio(filepath, target_sr=16000):
    """
    Loads an audio file and optionally resamples it to target_sr.
    """
    waveform, sample_rate = torchaudio.load(filepath)
    
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)
        sample_rate = target_sr
        
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    return waveform, sample_rate

def mel_delta_modulate(waveform, sample_rate=16000, n_mels=16, hop_length=160, threshold=5.0):
    """
    Converts a continuous waveform into a multi-channel sparse spike train using a 
    Mel-filterbank and Delta Modulation.
    
    Args:
        waveform (torch.Tensor): Shape (1, time)
        sample_rate (int): Audio sample rate.
        n_mels (int): Number of Mel frequency bins.
        hop_length (int): Hop length for the Mel spectrogram (determines time steps).
                          160 at 16kHz = 10ms per time step.
        threshold (float): Difference threshold (in dB) to trigger a spike in a frequency bin.
        
    Returns:
        spikes (torch.Tensor): Spike train of shape (time_steps, n_mels * 2) 
                               where channels are (bin0_UP, bin0_DOWN, bin1_UP, bin1_DOWN, ...)
    """
    # 1. Compute Mel Spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=400,
        hop_length=hop_length
    )
    
    # Shape: (1, n_mels, time_frames) -> (n_mels, time_frames)
    mel_spec = mel_transform(waveform).squeeze(0)
    
    # Convert to log scale for more perceptually uniform changes (dB scale)
    log_mel = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    
    n_bins, time_frames = log_mel.shape
    
    # Initialize spike train: [time_frames, n_bins * 2]
    spikes = torch.zeros((time_frames, n_bins * 2), dtype=torch.float32)
    
    # 2. Apply Delta Modulation across time for each frequency bin
    for t in range(1, time_frames):
        # Difference from previous frame
        diff = log_mel[:, t] - log_mel[:, t-1]
        
        # UP spikes (positive change > threshold)
        up_idx = diff >= threshold
        # DOWN spikes (negative change < -threshold)
        down_idx = diff <= -threshold
        
        # Interleave UP and DOWN channels:
        # Channel 0: bin0_UP, Channel 1: bin0_DOWN, Channel 2: bin1_UP...
        spikes[t, 0::2][up_idx] = 1.0
        spikes[t, 1::2][down_idx] = 1.0
        
    return spikes

def plot_mel_spikes(waveform, spikes, sample_rate, hop_length=160, title="Mel-filterbank Spikes"):
    """
    Visualizes the multi-channel spike train generated from the Mel-filterbank.
    """
    waveform = waveform.squeeze().numpy()
    spikes = spikes.numpy()
    
    time_frames = spikes.shape[0]
    n_channels = spikes.shape[1]
    n_mels = n_channels // 2
    
    # Time axes
    time_axis_wav = np.arange(len(waveform)) / sample_rate
    time_axis_spk = np.arange(time_frames) * (hop_length / sample_rate)
    
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot waveform
    axs[0].plot(time_axis_wav, waveform, color='b', alpha=0.7)
    axs[0].set_title(f"{title} - Raw Waveform")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid(True)
    
    # Plot multi-channel spikes
    # For a dense event plot, we plot each channel as a separate row (y-axis = channel)
    up_spikes = []
    down_spikes = []
    
    for c in range(n_mels):
        up_times = time_axis_spk[spikes[:, c*2] == 1.0]
        down_times = time_axis_spk[spikes[:, c*2 + 1] == 1.0]
        up_spikes.append(up_times)
        down_spikes.append(down_times)
        
    axs[1].eventplot(up_spikes, colors='g', lineoffsets=np.arange(n_mels), linelengths=0.8, alpha=0.7)
    axs[1].eventplot(down_spikes, colors='r', lineoffsets=np.arange(n_mels), linelengths=0.8, alpha=0.7)
    
    axs[1].set_title("Multi-channel Spike Train (Green=UP, Red=DOWN)")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Mel Frequency Bin")
    axs[1].set_yticks(np.arange(0, n_mels, 4))
    axs[1].grid(True)
    
    # Custom legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='g', lw=2), Line2D([0], [0], color='r', lw=2)]
    axs[1].legend(custom_lines, ['UP Spikes (Energy Increase)', 'DOWN Spikes (Energy Decrease)'], loc='upper right')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test script with synthetic waveform
    print("Running multi-channel audio_encoder test...")
    sample_rate = 16000
    t = np.linspace(0, 1, sample_rate) # 1 second
    
    # Create a waveform: silence (0.2s), low freq sine (0.3s), silence (0.2s), high freq sine (0.3s)
    waveform_np = np.zeros_like(t)
    waveform_np[int(0.2*sample_rate):int(0.5*sample_rate)] = 0.5 * np.sin(2 * np.pi * 300 * t[int(0.2*sample_rate):int(0.5*sample_rate)])
    waveform_np[int(0.7*sample_rate):int(1.0*sample_rate)] = 0.5 * np.sin(2 * np.pi * 3000 * t[int(0.7*sample_rate):int(1.0*sample_rate)])
    
    # Add minor noise everywhere
    waveform_np += np.random.normal(0, 0.01, size=t.shape)
    
    waveform = torch.tensor(waveform_np, dtype=torch.float32).unsqueeze(0)
    
    n_mels = 16
    # With AmplitudeToDB, threshold usually needs to be larger, e.g., 5.0 to 10.0 dB change
    threshold = 5.0 
    
    print(f"Encoding waveform... Mel bins: {n_mels}, Threshold: {threshold}dB")
    spikes = mel_delta_modulate(waveform, sample_rate=sample_rate, n_mels=n_mels, threshold=threshold)
    
    print(f"Spike tensor shape (time_steps, channels): {spikes.shape}")
    print(f"Total UP spikes: {spikes[:, 0::2].sum().item()}")
    print(f"Total DOWN spikes: {spikes[:, 1::2].sum().item()}")
    
    plot_mel_spikes(waveform, spikes, sample_rate, title="Synthetic Frequency Test")
