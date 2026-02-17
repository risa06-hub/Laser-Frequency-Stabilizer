import numpy as np

class self_heterodyne:
    def __init__(self, fiber_delay=1e-6,aom_freq=80e6,split_ratio=0.5,fiber_loss=0.1):
        #fiber delay= time delay in fibre
        #split ration= beam splitter ration (fraction to AOM path)
        #fiber loss= power loss in fiber

        self.fiber_delay= fiber_delay
        self.aom_freq=aom_freq
        self.split_ratio= split_ratio
        self.fiber_loss= fiber_loss

        self.aom_omega=2*np.pi*aom_freq
        self.fiber_buffer= None

    def process(self, laser_field, time, sample_rate, add_noise=True):

        n_samples=len(laser_field)
        dt=1/sample_rate
        print("SELF HETERODYNE PROCESSING")
        print(f"Fiber delay: {self.fiber_delay*1e6:.2f}microsec" )
        print(f"AOM freq: {self.aom_freq/1e6:.1f}MHz")
        print(f"Beam splitter ratio: {self.split_ratio:.2f}")

        #Beam splitting
        amplitude=np.abs(laser_field[0])
        path1=np.sqrt(self.split_ratio)*laser_field  #to AOM
        path2=np.sqrt(1- self.split_ratio)*laser_field #to Fiber

        #AOM
        path1= path1*np.exp(1j*self.aom_omega*time)

        print(f"added frequency shift")

        #fiber path
        delay_samples=int(self.fiber_delay * sample_rate)

        if delay_samples>0:
            #using a buffer
            path2_delayed=np.zeros_like(path2, dtype=complex)
            path2_delayed[delay_samples:]=path2[:-delay_samples]  #path_delayed[n]=path[n-dalay]

            #add some phase noise in the fiber
            fiber_phase_noise=0.05*np.random.randn(n_samples)
            path2_delayed=path2_delayed*np.exp(1j*fiber_phase_noise)

            #apply fiber loss due to power attenuation, here P_out=(1-L)P_in. since P=(Amp)^2, so,
            #A_out=root(1-L)A_in

            path2_delayed=path2_delayed*np.sqrt(1-self.fiber_loss)
        else:
            path2_delayed=path2.copy()

        #Combine at photodiode(Interference)
        print("COmbining beams at photodiode")
        combined_field= path1 + path2_delayed

        #photodiode intensity= |E|^2
        intensity=np.abs(combined_field)**2
        print(f"combined field power: {np.mean(intensity):.3f}W")

        #Conversion to voltage
        responsivity=0.8 #A/W
        load_resistor=50 #ohms

        photocurrent= intensity*responsivity
        photovoltage=photocurrent*load_resistor

        print(f"mean photvoltage: {np.mean(photovoltage):.3f}V")
        print(f"AC amplitude: {np.std(photovoltage):.3f}V")

        #Adding electronic noise
        if add_noise:
            print("Adding electronic noise")

            #shot nosie( sqrt(power))
            shot_noise_std=1e-3*np.sqrt(np.mean(photovoltage))
            shot_noise=shot_noise_std*np.random.randn(n_samples)

            #Thermal noise(Johnson)
            thermal_noise_std=5e-4 #0.5mV rms
            thermal_noise=thermal_noise_std*np.random.randn(n_samples)

            photovoltage_noisy= photovoltage + shot_noise + thermal_noise

            print(f"Shot noise: {shot_noise_std:.2e} V RMS")
            print(f"Thermal noise: {thermal_noise_std:.2e} V RMS")
            print(f"Total noise: {np.std(shot_noise+thermal_noise):.2e} V RMS")

            photovoltage = photovoltage_noisy

        return photovoltage, path1, path2_delayed

    def get_phase_difference(self,path1,path2):
        phase1=np.angle(path1)
        phase2=np.angle(path2)
        phase_diff= phase1 - phase2

        #remove 2pi jumps
        phase_diff=np.unwrap(phase_diff)
        return phase_diff
