import numpy as np

class simple_laser:

    def __init__(self, wavelength=2e-6, power=1.0, linewidth=10e3, flicker_level=1e6,white_phase_level=1e-12):
        self.wavelength= wavelength
        self.power=power
        self.linewidth= linewidth

        self.c=299792458
        self.frequency= self.c/wavelength
        self.omega=2*np.pi*self.frequency

        self.h0= linewidth/np.pi #for whtie nosie, delta_V= pi*h0
        self.h1=flicker_level
        self.h2=white_phase_level
        self.phase_history=None
        self.field_history=None
        self.time_array= None


    def genrate_1f_noise(self,n_samples,alpha=1):
        freqs=np.fft.rfftfreq(n_samples)

        #to avoid division by zero at DC:
        spectrum=np.ones(len(freqs),dtype=complex)
        mask=freqs>0
        spectrum[mask]=1/(freqs[mask]**(alpha/2))
        random_phase=np.exp(1j*2*np.pi*np.random.rand(len(spectrum)))
        spectrum=spectrum*random_phase

        #ifft to get the time domain
        noise=np.fft.irfft(spectrum,n_samples)

        #removing the DC componenet, f=0 since it has infinite power , which will produce an unbound mean and make the time domain signal meaningless

        noise=noise-np.mean(noise)
        noise=noise/np.std(noise) #to get controled signal power and fair comparison across the parameters

        return noise

    def generate_field(self,duration,sample_rate):

        #time array
        n_samples= int(duration*sample_rate)
        self.time_array= np.arange(n_samples)/sample_rate
        dt=1/sample_rate

        print("generatig laser field:")
        print(f"Duration: {duration*1e6:.1f}microsec")
        print(f"Samples: {n_samples}")
        print(f"sampling rate: {sample_rate/1e6:.1f}MHz")
        print(f"laser freq: {self.frequency/1e12:.2f}THz")
        print(f"linewidth: {self.linewidth/1e3:.1f}kHz")

        #noise modeling, white freq noise
        freq_noise_std= np.sqrt(self.h0/dt)
        white_freq_noise= freq_noise_std*np.random.randn(n_samples)

        #to get the phase
        phase_white_noise= 2*np.pi*np.cumsum(white_freq_noise)*dt     #intergration, returns array where each element is the sum of all the previous one

        #flicker nosie
        flicker_noise= self.genrate_1f_noise(n_samples,alpha=1)
        #flicker noise grows logarrithmic and psd is the variance, so std is given by sqrt
        flicker_scale=np.sqrt(self.h1*np.log(sample_rate/0.1)*dt) # 1/f noise= var= log(duration)
        phase_flicker= flicker_scale*flicker_noise

        #white phase nosie, small
        phase_std=np.sqrt(self.h2/dt)
        phase_white= phase_std*np.random.randn(n_samples)

        #total phase
        phase_total=phase_white_noise+phase_flicker+phase_white

        self.phase_components={'white_freq': phase_white_noise,
                               'flicker':phase_flicker,
                               'white_phase':phase_white,
                               'total':phase_total}

        #GENERATE ELECTRIC FIELD
        carrier_phase= self.omega*self.time_array
        total_phase= carrier_phase + phase_total
        amplitude=np.sqrt(self.power)
        field=amplitude*np.exp(1j*total_phase)

        self.phase_history=phase_total
        self.field_history=field

        return self.time_array, field, phase_total

