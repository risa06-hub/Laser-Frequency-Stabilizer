import numpy as np
import matplotlib.pyplot as plt

class demodulator:

    def __init__(self,aom_freq=80e6, lowpass_cutoff-10e6):
        self.aom_freq=aom_freq
        self.omega_aom= 2*np.pi*aom_freq
        self.lowpass_cutoff=lowpass_cutoff

    def demodulate(self,photovoltage, time, sample_rate):

        n_samples=len(photovoltage)
        dt=1/sample_rate

        #reference signal
        ref_I=np.cos(self.omega_aom*time)
        ref_Q=np.sin(self.omega_aom*time)

        #mixing
        mixed_I=photovoltage*ref_I
        mixed_Q=photovoltage*ref_Q

        #lowpass filtering
        print("Applying lowpass filter")
        #cut off approx= f_c=0.433/N*dt= lowpass lowpass_cutoff
        window_size=int(0.433/self.lowpass_cutoff*dt)
        window_size=max(5,min(window_size,50))

        #Applying Moving Average
        kernel=np.ones(window_size) #np.ones creates an array of size window_size with values 1, so that the weights sum to 1. this is the h[n]=impulse response

        I_filtered=np.convolve(mixed_I,kernel, mode='same')
        Q_filtered=np.convolve(mixed_Q,kernel, mode='same')

        #extract phase error
        phase_error= np.arctan2(Q_filtered,I_filtered)
        phase_error=phase_error-np.mean(phase_error) #to remove DC offset

        return phase_error, I_filtered,Q_filtered

    def plot(self, time, photovoltage, phase_error):
        fig, (ax1,ax2)=plt.subplots(1,2,figsize=(12,4))
        t_us=time*1e6

        #1st plot
        ax1.plot(t_us,photovoltage,'-b',linewidth=0.8)
        ax1.set_xlabel('Time(us)')
        ax1.set_ylabel('Voltage(V)')
        ax1.set_title('Raw Photodiode signal(80MHz carrier)')
        ax1.grid(True,alpha=0.3)

        #Adding zoom for first 200 samples
        if len(t_us)>200:
            ax1_insert=ax1.insert([0.55,0.55,0.4,0.4])
            ax1_inset.plot(t_us[:200], photovoltage[:200], 'b-', linewidth=1)
            ax1_inset.set_xlabel('Time (μs)')
            ax1_inset.set_ylabel('Voltage (V)')
            ax1_inset.set_title('Zoom: 80 MHz')
            ax1_inset.grid(True, alpha=0.3)

        #2nd plot to get the phase error
        ax2.plot(t_us,phase_error,'r-', lw=1)
        ax2.set_xlabel('Time(us)')
        ax2.set_ylabel('Phase error(rad)')
        ax2.set_title("Extracted Phase error: phi(t)- phi(t-T)")
        ax2.grid(True, alpha=0.3)

        #to get the statistic box
        stats_text=f'RMS:{np.std(phase_error):.3e}rad\n Peak:{np.max(np.abs(phase_error)):.3e}rad'
        ax2.test(0.05,0.95,stats_text,transform=ax2.transAxes,verticalalignmnet='top',bbox=dict(boxstyle='round', facecolor='wheat',alpha=0.8))

        plt.tight_layout()
        plt.show()

         # Print summary
        print(f"\nSummary:")
        print(f"  Raw signal: {np.mean(photovoltage):.1f} V mean, {np.std(photovoltage):.1f} V RMS")
        print(f"  Phase error: {np.std(phase_error):.3e} rad RMS")
        print(f"  That's {np.std(phase_error)/(2*np.pi):.1e} fringes RMS")

    def plot_IQ(self,time,I,Q):

        fig, (ax1,ax2)=plt.subplots(1,2,figsize=(12,4))
        t_us=time*1e6

        #I and Q components
        ax1.plot(t_us,I,'b-',alpha=0.7)
        ax1.plot(t_us,Q,'g-',alpha=0.7)
        ax1.set_xlabel('Time(us)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('I and Q components after Filter')
        ax1.grid(True,alpha=0.3)

        #IQ
        ax2.plot(I[:200],Q[:200],'b-',alpha=0.6,markersize=3)
        ax2.set_xlabel('I')
        ax2.set_ylabel('Q')
        ax2.set_title('IQ Constellation')
        ax2.axis('equal')

        plt.tight_layout()
        plt.show()


