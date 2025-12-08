import numpy as np
import matplotlib.pyplot as plt

# Laser parameters
wavelength= 2.05e-6
c = 3e8
f0 = c / wavelength  # Center frequency ≈ 146.341 THz

# AOM (Acousto-Optic Modulator)
f_AOM = 80e6  # 80 MHz frequency shift

# Delay line (optical fiber)
t_delay = 0.10e-6  # 0.10 microsec time delay (~20m fiber)

# Photodetector
R_gain = 1e3  # Transimpedance gain (V/A)
V_noise_rms = 0.01  # RMS detector noise (V)

# Simulation parameters
fs = 500e6  # Sampling rate = 500 MHz
N = 10000  # Number of samples (EVEN number)
T = N / fs  # Simulation time = 20 microsec

# PSD parameters
h_m2 = 1e-2   # rad²·Hz for 1/f²
h_m1 = 1e-4   # rad² for 1/f
h_0 = 1e-6    # rad²/Hz for white frequency noise
h_1 = 1e-12   # rad² for f¹
h_2 = 1e-18   # rad²/Hz² for f²

print(f"Generating {N} samples at {fs/1e6:.0f} MHz sampling rate")
print(f"Time delay = {t_delay*1e6:.2f} microsec")
print(f"Total simulation time: {T*1e6:.2f} microsec")
print(f"N is even: {N % 2 == 0}")


def generate_phase_noise_fft(N, fs, h_m2, h_m1, h_0, h_1, h_2):

    #frequency array
    f = np.fft.fftfreq(N, 1/fs)

    #PSD with absolute frequencies
    f_abs = np.abs(f)
    f_abs[0] = 1e-15  # Avoid division by zero

    #PSD model
    S_phi = h_m2/(f_abs**2) + h_m1/f_abs + h_0 + h_1*f_abs + h_2*(f_abs**2)

    # Frequency resolution
    delta_f = fs / N

    #Hermitian symmetric spectrum
    X = np.zeros(N, dtype=complex)

    # Since N=10000 is even, we have:
    # half_N = N//2 = 5000
    # Indices: 0 (DC), 1-4999 (positive), 5000 (Nyquist), 5001-9999 (negative)

    half_N = N // 2  # 5000

    # Generating random phases for positive frequencies including DC
    # We need half_N + 1 = 5001 values (indices 0 to 5000)
    random_phases = np.exp(1j * 2 * np.pi * np.random.rand(half_N + 1))

    # Magnitudes for DC to Nyquist (indices 0 to 5000)
    magnitudes = np.sqrt(S_phi[:half_N + 1] * delta_f)

    # Assign to first half (DC to Nyquist)
    X[:half_N + 1] = magnitudes * random_phases

    # Apply Hermitian symmetry for negative frequencies
    # For indices 5001 to 9999, they are complex conjugates of indices 4999 down to 1
    # X[5001]= X*[4999], X[5002]= X*[4998], ..., X[9999] = X*[1]

    # negative frequency components (indices N-1 down to half_N+1)
    for k in range(1, half_N):
        X[N - k] = np.conj(X[k])

    # DC and Nyquist are real
    X[0] = np.real(X[0])  # DC component real
    X[half_N] = np.real(X[half_N])  # Nyquist component real (index 5000)

    #Inverse FFT
    phi_t = np.real(np.fft.ifft(X))

    #Normalize
    phi_t = 0.1 * phi_t / np.std(phi_t)

    return phi_t, f, S_phi

# Generate phase noise
phi_t, f_freq, S_phi_psd = generate_phase_noise_fft(N, fs, h_m2, h_m1, h_0, h_1, h_2)
t = np.arange(N) / fs

print(f"Phase noise generated successfully!")
print(f"Phase noise shape: {phi_t.shape}")
print(f"Phase noise mean: {np.mean(phi_t):.3e}, std: {np.std(phi_t):.3e}")

#Original laser field
E0 = np.exp(1j * (2*np.pi*f0*t + phi_t))

#After beam splitter (50/50)
E_AOM = (1/np.sqrt(2)) * np.exp(1j * (2*np.pi*(f0 + f_AOM)*t + phi_t))

#Delayed path
delay_samples = int(t_delay * fs)
print(f"Delay in samples: {delay_samples}")

#delayed phase
phi_delayed = np.zeros_like(phi_t)
if delay_samples > 0:
    phi_delayed[delay_samples:] = phi_t[:-delay_samples]
    phi_delayed[:delay_samples] = phi_t[0]

E_delay = (1/np.sqrt(2)) * np.exp(1j * (2*np.pi*f0*(t - t_delay) + phi_delayed))

#Interference at photodiode
E_total = E_AOM + E_delay

#Photodetector voltage
I_pd = np.abs(E_total)**2
V_pd = R_gain * I_pd + np.random.normal(0, V_noise_rms, N)

#phase difference
delta_phi = phi_t - phi_delayed

#Plotting
plt.figure(figsize=(12, 8))

# Plot 1: Photodetector voltage (first 2 microsec)
plt.subplot(2, 2, 1)
plt.plot(t*1e6, V_pd, 'r-', linewidth=0.8, alpha=0.8)
plt.xlabel('Time (microsec)')
plt.ylabel('Voltage (V)')
plt.title('Photodetector Voltage vs Time (0-2 microsec)')
plt.grid(True, alpha=0.3)
plt.xlim([0, 2])

# Plot 2: Zoomed to show 80 MHz carrier
plt.subplot(2, 2, 2)
zoom_samples = min(int(0.1e-6 * fs), N)
plt.plot(t[:zoom_samples]*1e9, V_pd[:zoom_samples], 'r-', linewidth=1.5)
plt.xlabel('Time (microsec)')
plt.ylabel('Voltage (V)')
plt.title('Zoom: 80 MHz Carrier (0-100 microsec)')
plt.grid(True, alpha=0.3)

# Add expected 80 MHz periods
period_ns = 1/f_AOM * 1e9  # 12.5 ns
for i in range(1, 9):
    plt.axvline(i*period_ns, color='k', linestyle=':', alpha=0.3, linewidth=0.5)

# Plot 3: Phase noise
plt.subplot(2, 2, 3)
plt.plot(t*1e6, phi_t, 'b-', linewidth=0.8)
plt.xlabel('Time (microsec)')
plt.ylabel('Phase (rad)')
plt.title('Laser Phase Noise')
plt.grid(True, alpha=0.3)
plt.xlim([0, 2])

# Plot 4: Phase difference
plt.subplot(2, 2, 4)
plt.plot(t*1e6, delta_phi, 'purple', linewidth=0.8)
plt.xlabel('Time (microsec)')
plt.ylabel('delta_phi (rad)')
plt.title(f'Phase Difference: φ(t) - φ(t-{t_delay*1e6:.2f}microsec)')
plt.grid(True, alpha=0.3)
plt.xlim([0, 2])

plt.tight_layout()
plt.show()

#summary
print("\n" + "*"*60)
print("SIMULATION SUMMARY")
print("*"*60)
print(f"Mean voltage: {np.mean(V_pd):.3f} V")
print(f"Voltage RMS: {np.std(V_pd):.3f} V")
print(f"Voltage range: [{np.min(V_pd):.3f}, {np.max(V_pd):.3f}] V")
print(f"Phase noise RMS: {np.std(phi_t):.4f} rad")
print(f"Phase diff RMS: {np.std(delta_phi):.4f} rad")
print(f"80 MHz period: {1/f_AOM*1e9:.2f} ns")
print(f"Delay τ: {τ*1e9:.1f} ns ({delay_samples} samples)")
print("="*60)

# Show first 10 voltage values
print("\nFirst 10 voltage-time pairs:")
print("Time (ns)  |  Voltage (V)")
print("-" * 30)
for i in range(min(10, N)):
    print(f"{t[i]*1e9:8.2f}  |  {V_pd[i]:8.4f}")
