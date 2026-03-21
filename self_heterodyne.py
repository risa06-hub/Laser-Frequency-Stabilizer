import numpy as np

class SelfHeterodyne:

    def __init__(self, fiber_delay=1e-6, aom_freq=80e6,split_ratio=0.5, fiber_loss=0.1, fiber_phase_noise_std=0.05):
        self.fiber_delay= fiber_delay
        self.aom_freq = aom_freq
        self.split_ratio= split_ratio
        self.fiber_loss = fiber_loss
        self.fiber_phase_noise_std = fiber_phase_noise_std   # rad RMS

        self.aom_omega = 2 * np.pi * aom_freq

    def process(self, laser_field, time, sample_rate, add_noise=True):
        
        n_samples = len(laser_field)

        print("Self-heterodyne processing:")
        print(f"  Fiber delay  : {self.fiber_delay*1e6:.2f} us")
        print(f"  AOM freq     : {self.aom_freq/1e6:.1f} MHz")
        print(f"  Split ratio  : {self.split_ratio:.2f}")

        # Beam splitter
        path1 = np.sqrt(self.split_ratio) * laser_field   #AOM
        path2 = np.sqrt(1.0 - self.split_ratio)* laser_field   #fiber

        #AOM
        path1 = path1 * np.exp(1j * self.aom_omega * time)
        print(" +AOM frequency shift applied")

        #Fiber delay
        delay_samples = int(self.fiber_delay * sample_rate)
        path2_delayed = np.zeros(n_samples, dtype=complex)
        if delay_samples > 0:
            path2_delayed[delay_samples:] = path2[:-delay_samples]
        else:
            path2_delayed = path2.copy()

        # Fiber environmental phase noise (white)
        fiber_pnoise = self.fiber_phase_noise_std * np.random.randn(n_samples)
        path2_delayed *= np.exp(1j * fiber_pnoise)

        # Power attenuation: Power loss in the fiber
        path2_delayed *= np.sqrt(1.0 - self.fiber_loss)

        #Photodiode
        combined = path1 + path2_delayed
        intensity = np.abs(combined) ** 2 

        responsivity= 0.8    # A/W
        load_resistor= 50.0   # Ω
        photovoltage= intensity*responsivity*load_resistor

        print(f"Mean photovoltage : {np.mean(photovoltage):.3f} V")
        print(f"AC RMS            : {np.std(photovoltage):.3f} V")

        #Electronic noise
        if add_noise:
            # Shot noise
            shot_std= 1e-3 * np.sqrt(np.mean(photovoltage))
            thermal_std= 5e-4   #(Johnson noise at room temp, 50 Ω)

            shot_noise = shot_std * np.random.randn(n_samples)
            thermal_noise = thermal_std * np.random.randn(n_samples)
            photovoltage = photovoltage + shot_noise + thermal_noise

            print(f"Shot noise    : {shot_std:.2e} V RMS")
            print(f"Thermal noise : {thermal_std:.2e} V RMS")

        return photovoltage, path1, path2_delayed

    def get_phase_difference(self, path1, path2):

        phase_diff = np.angle(path1) - np.angle(path2)
        return np.unwrap(phase_diff)
