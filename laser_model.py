import numpy as np

class SimpleLaser:
    
    def __init__(self, wavelength=2e-6, power=1.0, linewidth=10e3, flicker_level=1e6, white_phase_level=1e-12):
        self.wavelength = wavelength
        self.power = power
        self.linewidth = linewidth

        self.c = 299_792_458
        self.frequency = self.c / wavelength
        self.omega = 2 * np.pi * self.frequency

        # PSD coefficients (S_ν(f) = h0 + h1/f + h2·f^2)
        self.h0 = linewidth / np.pi      # white noise:delta ν = pi·h0
        self.h1 = flicker_level          # 1/f FM
        self.h2 = white_phase_level      # white PM

        self.phase_history = None
        self.field_history = None
        self.time_array = None

    
    def _generate_1f_noise(self, n_samples, alpha=1.0):
        
        freqs = np.fft.rfftfreq(n_samples)          # [0,1/(2dt)] normalised
        spectrum = np.ones(len(freqs), dtype=complex)
        mask = freqs > 0
        spectrum[mask] = 1.0 / (freqs[mask] ** (alpha / 2.0))
        spectrum[0] = 0.0                            # kill DC component

        spectrum *= np.exp(1j * 2 * np.pi * np.random.rand(len(spectrum)))
        noise = np.fft.irfft(spectrum, n_samples)
        noise -= noise.mean()
        noise /= noise.std()                         
        return noise


    def generate_field(self, duration, sample_rate):
        n_samples = int(duration * sample_rate)
        dt = 1.0 / sample_rate
        self.time_array = np.arange(n_samples) / sample_rate

        print("Generating laser field:")
        print(f"  Duration      : {duration*1e6:.1f} us")
        print(f"  Samples       : {n_samples}")
        print(f"  Sample rate   : {sample_rate/1e6:.1f} MHz")
        print(f"  Laser freq    : {self.frequency/1e12:.4f} THz")
        print(f"  Linewidth     : {self.linewidth/1e3:.1f} kHz")

        # White frequency noise 
        # Var(delt ν per bin) = h0/dt
        # std = sqrt(h0/dt)
        freq_noise_std = np.sqrt(self.h0 / dt)
        white_freq_noise = freq_noise_std * np.random.randn(n_samples)
        phase_white_fm = 2 * np.pi * np.cumsum(white_freq_noise) * dt

        # Flicker (1/f) frequency noise 
        # Variance of 1/f phase ≈ h1 * ln(N) * dt  (log-divergent process)
        flicker_noise = self._generate_1f_noise(n_samples, alpha=1.0)
        flicker_scale = np.sqrt(self.h1 * np.log(max(n_samples, 2)) * dt)
        phase_flicker = flicker_scale * flicker_noise

        # White phase noise 
        phase_wn_std = np.sqrt(self.h2 / dt)
        phase_white_pm = phase_wn_std * np.random.randn(n_samples)

        # Total phase 
        phase_total = phase_white_fm + phase_flicker + phase_white_pm

        self.phase_components = {
            'white_freq' : phase_white_fm,
            'flicker'    : phase_flicker,
            'white_phase': phase_white_pm,
            'total'      : phase_total,
        }

        # electric field 
        carrier_phase = self.omega * self.time_array
        total_phase   = carrier_phase + phase_total
        amplitude     = np.sqrt(self.power)
        field         = amplitude * np.exp(1j * total_phase)

        self.phase_history = phase_total
        self.field_history = field

        return self.time_array, field, phase_total


