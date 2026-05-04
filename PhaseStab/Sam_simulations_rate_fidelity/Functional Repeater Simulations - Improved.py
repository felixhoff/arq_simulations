# -*- coding: utf-8 -*-
"""
Functional Repeater Simulation:
Estimate coincidence rates between photon events in a quantum memory link.

Created by: sgrandi
"""

import numpy as np
import matplotlib.pyplot as plt
import numba as nb
from scipy import signal
from itertools import accumulate
import statistics
from matplotlib.colors import LinearSegmentedColormap

# Close any previous plots
plt.close('all')

# --------------------------------------
# Custom Color Palette
# --------------------------------------
yellow = (255/255, 159/255, 14/255)
yellow_2 = (255/255, 72/255, 24/255)
yellow_3 = (251/255, 218/255, 19/255)
yellow_d = (255/255, 122/255, 35/255)
brown = (187/255, 59/255, 14/255)

purple = (193/255, 51/255, 255/255)
purple_2 = (220/255, 143/255, 255/255)
purple_3 = (153/255, 0/255, 222/255)
purple_d = (113/255, 0/255, 164/255)

# Custom colormap for plots
colors = [(16/255, 0/255, 134/255), (192/255, 183/255, 255/255)]
cmap = LinearSegmentedColormap.from_list("Custom", colors, N=4)


@nb.njit()
def learn_statistics_either_var2(trig_1, trig_2, tout, lat):
    """
    Match heralding events across two channels with a bounded delay window.

    Parameters
    ----------
    trig_1 : array-like
        Time tags of heralding events from channel 1.
    trig_2 : array-like
        Time tags of heralding events from channel 2.
    tout : float
        Maximum allowable delay between two correlated events [s].
    lat : float
        Latency after a successful event before memory is ready again [s].

    Returns
    -------
    her : list of float
        Heralded time tags from channel 1.
    cl : list of float
        Correlated time tags from channel 2.
    ev_list : list of int
        Index of which event was earliest (0 or 1).
    del_list : list of float
        Time differences between matched events.
    c_tout : int
        Count of discarded events due to timeout.
    c_mark : int
        Count of discarded events due to memory not ready.
    c_ok : int
        Count of successfully matched events.
    j : int
        Index reached in iteration.
    """
    
    her = []
    cl = []
    del_list = []
    ev_list = []

    c_tout = 0  # timeout count
    c_ok = 0    # matched count
    c_mark = 0  # memory-not-ready count

    len_com = min(len(trig_1), len(trig_2))
    t_mark = 0  # time when memory is next available
    j = 0

    while j < len_com:
        vec = np.array([trig_1[j], trig_2[j]])
        j_min = int(np.argmin(vec))  # earliest event
        j_max = int(np.argmax(vec))  # later event
        t_ref = np.float64(vec[j_min])
        t_count = np.float64(vec[j_max])

        # Skip if memory isn't ready
        if t_ref <= t_mark:
            if j_min == 0:
                trig_1 = np.delete(trig_1, j)
            else:
                trig_2 = np.delete(trig_2, j)
            c_mark += 1
            len_com -= 1
        else:
            # If the second event is too late, discard the earlier one
            if t_count >= t_ref + tout:
                if j_min == 0:
                    trig_1 = np.delete(trig_1, j)
                else:
                    trig_2 = np.delete(trig_2, j)
                c_tout += 1
                len_com -= 1
                t_mark = t_ref + tout
            else:
                # Successful match!
                her.append(t_ref)
                cl.append(t_count)
                ev_list.append(j_min)

                # Calculate delay between matched events
                t_diff = int(t_count * 1e7) - int(t_ref * 1e7)
                del_list.append(t_diff * 1e-7)

                t_mark = t_count + lat
                c_ok += 1
                j += 1

    return her, cl, ev_list, del_list, c_tout, c_mark, c_ok, j


#%%
"""
SPECS

details for both source and memory
"""

# Select parameter set: 0 = current SW, 1 = current DD, 2 = optimistic, 3 = very optimistic, 4 = Welinq
par = 1


# ------------------------------
# Quantum Memory Parameters
# ------------------------------
tau_AFC = 1e-6                            # AFC storage time [s]
tau_SW = np.linspace(0.1, 20.1, 100) * 1e-3  # Spin-wave storage time range [s]
tau_min = 80e-6                            # Minimum total storage time [s]

eta_AFC0 = [0.3, 0.6, 0.7, 0.9, 0.9][par]       # AFC efficiency at zero delay
eta_CP = [0.7, 0.8, 0.9, 0.98, 1][par]        # Control pulse efficiency

cycle_mem = 1                              # Memory cycle duration [s]
prep_mem = [0.35, 0.3, 0.25, 0.2, 0.1][par]     # Time needed to reinitialize memory [s]
eta_duty_mem = 1 - prep_mem / cycle_mem    # Duty cycle of the memory

eta_duty_chopper = [20/33, 20/33, 22/33, 25/33, 1][par]  # Duty cycle of the chopper
eta_duty = eta_duty_mem * eta_duty_chopper           # Combined duty cycle

lat = 150e-6                               # Memory latency after a success [s]

NF = [8.2e-4, 8.2e-4, 6e-4, 5e-4, 5e-4][par]      # Noise floor
T2 = 115e-6                                # AFC effective coherence time [s]
gamma_inhom = 14e3                         # Inhomogeneous spin broadening [Hz]
T_eff = [2e-3, 4e-3, 8e-3, 20e-3, 20e-3][par]      # Effective memory lifetime with DD [s]
M_m = 2.2                                  # DD model exponent (empirical)


# ------------------------------
# SPDC Source Parameters
# ------------------------------
etaH = [0.3, 0.5, 0.6, 0.8, 0.6][par]           # Heralding efficiency
P = 0.4                                   # SPDC pump power [mW]
a = 6                                      # g²(0) model parameter
bright = 1_700_000                         # Brightness [counts/s/mW]
M = 15                                     # Multimode scaling factor
Rid0 = 2 * bright / M                      # Heralding rate at source output [Hz/mW]

g2 = 1 + a / P                             # Cross-correlation function


# ------------------------------
# Sync & Timing Capabilities
# ------------------------------
dt = 600e-9                                # Time bin resolution [s]
t_out = tau_AFC + tau_SW                   # Max waiting time for a match [s]


# ------------------------------
# Optical Transmission Efficiencies
# ------------------------------
eta_T1 = [0.6, 0.7, 0.8, 0.95, 1][par]        # Source-to-memory
eta_T2 = [0.5, 0.6, 0.8, 0.90, 0.9][par]        # Memory-to-filter
eta_T3 = [0.7, 0.8, 0.90, 0.95, 1][par]       # Filter-to-fiber


# ------------------------------
# Detection and Visibility
# ------------------------------
Vphase = [0.9, 0.95, 0.97, 0.96, 0.96][par]     # Phase visibility
etaOv = [0.9, 0.9, 0.95, 0.99, 0.99][par]        # Mode overlap

etaD = [0.85, 0.85, 0.90, 0.95, 0.9][par]       # Detector efficiency
etaDi = etaD                               # Idler detector
etaDs = [0.85, 0.85, 0.90, 0.95, 0.9][par]      # Signal detector


# ------------------------------
# Fiber Channel Specs
# ------------------------------
L = 50                                   # Fiber length [km]
alpha = 0.2                                # Attenuation [dB/km]
fib = 10 ** (-alpha * L / 10)              # Fiber transmission

c = 299_792.458                            # Speed of light [km/s]
n = 1.5                                    # Refractive index of fiber
t_comm = (2 * L * n) / c                   # Round-trip communication time [s]


# Fraction of time filled with useful modes
trial_frac = [1, 0.25, 0.5, 0.6, 0.1][par]


# ------------------------------
# Parameter override
# ------------------------------

# [eta_AFC0, 
#  eta_CP, 
#  etaH, 
#  trial_frac,
#  T_eff,
#  eta_T2,
#  Vphase] = [0.60, # AFC eff
#             0.80, # CP eff
#             0.50, # her eff
#             0.5, # frac modes
#             2e-3, # storage
#             0.9,
#             0.97] # transmission



# Deliverable D3.1 - par = 1
# ------------------------------
# D3.1
# [eta_duty, eta_AFC0, eta_CP, NF, etaH, trial_frac, T_eff, eta_T3, Vphase, etaOv] = [
#     0.42, 0.6, 0.9, 8.2e-4, 0.6, 0.25, 2e-3, 0.45/(eta_T1*eta_T2), 0.90, 0.95]

# D3.A
# [eta_duty, eta_AFC0, eta_CP, NF, etaH, trial_frac, T_eff, eta_T3, Vphase, etaOv] = [
#     0.42, 0.6, 0.8, 8.2e-4, 0.5, 0.25, 2e-3, 0.50/(eta_T1*eta_T2), 0.90, 0.95]


# Deliverable D3.2 - par = 2
# ------------------------------
# D3.2
# [eta_duty, eta_AFC0, eta_CP, NF, etaH, trial_frac, T_eff, eta_T3, Vphase, etaOv] = [
#     0.5, 0.95, 0.9, 6e-4, 0.6, 0.039, 8e-3, 0.58/(eta_T1*eta_T2), 0.96, 0.95]

# D3.B
# [eta_duty, eta_AFC0, eta_CP, NF, etaH, trial_frac, T_eff, eta_T3, Vphase, etaOv] = [
#     0.42, 0.6, 0.8, 8.2e-4, 0.5, 0.25, 2e-3, 0.50/(eta_T1*eta_T2), 0.90, 0.95]

# D3.b
# [eta_duty, eta_AFC0, eta_CP, NF, etaH, trial_frac, T_eff, eta_T3, Vphase, etaOv] = [
#     0.50, 0.65, 0.85, 8.2e-4, 0.5, 0.25, 1e-3, 0.58/(eta_T1*eta_T2), 0.92, 0.95]


# Deliverable D3.3 - par = 3 for D3.3, par = 2 for D3.C & D3.c
# ------------------------------
# D3.3
# [eta_duty, eta_AFC0, eta_CP, NF, etaH, trial_frac, T_eff, eta_T3, Vphase, etaOv] = [
#     0.61, 0.9, 0.98, 5e-4, 0.8, 0.6, 20e-3, 0.81/(eta_T1*eta_T2), 0.96, 0.99]

# D3.C
# [eta_duty, eta_AFC0, eta_CP, NF, etaH, trial_frac, T_eff, eta_T3, Vphase, etaOv] = [
#     0.50, 0.7, 0.9, 6e-4, 0.6, 0.6, 4e-3, 0.60/(eta_T1*eta_T2), 0.95, 0.97]

# D3.c
# [eta_duty, eta_AFC0, eta_CP, NF, etaH, trial_frac, T_eff, eta_T3, Vphase, etaOv] = [
#     0.50, 0.65, 0.85, 8.2e-4, 0.50, 0.5, 1e-3, 0.58/(eta_T1*eta_T2), 0.95, 0.97]

# D3.spatial multiplexing
# [eta_duty, eta_AFC0, eta_CP, NF, etaH, trial_frac, T_eff, eta_T3, Vphase, etaOv] = [
#     0.50, 0.35, 0.85, 8.2e-4, 0.50, 1, 8e-3, 0.58/(eta_T1*eta_T2), 0.95, 0.97]


#%%
"""
2-FOLD HERALDING RATE


MONTECARLO

Here I simulate the full experimental cycle, starting from the number of idlers
which are sent from the source, start simulating:
    1) First for a time given by the portion of the communication time which is covered
    by modes. When this is covered, skip to the next communication time.
    2) Repeat this until you have ran out of time and you have to prepare the memory again.
    3) Add the memory preparation time, and start over.
Finally, take all the successfull events in the two lists (still called Ctrigger and
Csignal, for historical reasons).


IMPORTANT
If the code breaks or gets stuck, reduce the number of repetitions.

"""

# Total time window to simulate per cycle
time_span = trial_frac * t_comm  # [s]

# Number of trials within a memory cycle (excluding prep time)
n_trials = int((cycle_mem - prep_mem) / t_comm)

# Repetitions for statistical averaging - almost equivalent to cryo cycles
repetition = 50

# Heralding rate (idler path), includes fiber loss and chopper duty
Rid = Rid0 * P * fib * etaDi * eta_duty_chopper

# Probability of detection within a time bin
pdt = Rid * dt

# Lists to store results
signal_list = []
idler_list = []

# Start time offset to skip memory prep window
time_mark = int(prep_mem / dt)

# Main simulation loop
for j in range(repetition):
    for i in range(n_trials):
        # Simulate random events for idler detections
        list_idler = np.random.rand(int(time_span / dt))
        index_idler = np.where((list_idler - pdt) < 0)
        idler_list.append(index_idler[0] + time_mark)

        # Simulate random events for signal detections
        list_signal = np.random.rand(int(time_span / dt))
        index_signal = np.where((list_signal - pdt) < 0)
        signal_list.append(index_signal[0] + time_mark)

        # Advance simulation time by one communication round
        time_mark += int(t_comm / dt)

        # Free memory
        del list_idler
        del list_signal

    # Add preparation window before next repetition
    time_mark += int(prep_mem / dt)

# Flatten event arrays and convert back to real time [s]
Ctrigger = np.array([item for sublist in idler_list for item in sublist], dtype=np.float64) * dt
Csignal = np.array([item for sublist in signal_list for item in sublist], dtype=np.float64) * dt

# Clip signals so both start and end at the same time window
tmin = max(min(Csignal), min(Ctrigger))
Csignal = Csignal[Csignal >= tmin]
Ctrigger = Ctrigger[Ctrigger >= tmin]

tmax = min(max(Csignal), max(Ctrigger))
Csignal = Csignal[Csignal <= tmax]
Ctrigger = Ctrigger[Ctrigger <= tmax]

# Measurement duration
tmeasure = tmax - tmin  # [s]

# Store length for future use
length_t = len(Ctrigger)
length_s = len(Csignal)

# # This plots the idlers, to show the correct identification of modes & communication time
# fig, ax = plt.subplots()
# ax.scatter(range(len(Ctrigger)), Ctrigger)
# [ax.axhline(n*cycle_mem, c = 'r', ls = '--') for n in range(3)]
# [ax.axhline(prep_mem + n*t_comm, c = 'k', ls = '--') for n in range(50)]
# # ax.set(xlabel = r'Delay (ms)', ylabel = 'Events')
# plt.tight_layout()
# plt.show()


#%%
"""
THE NUMERICAL BIT

Here I implement the experiment with the function 'learn_statistics_either_var2'. It has
a couple of loops inside, but the idea is:
    1) it takes the first element from both lists of counts, trig1 and trig2. These are our clicks
    to consider. It assigns variables for min and max, and keeps track of which is which.
    2) t_mark sets the time at which the memory is ready. At the beginning
    is zero, but after a successful event it will be that last timestamp plus a
    possible latency (you have to wait for the fluorescence). If the min is smaller than t_mark, 
    then discard it from its list, without advancing the other one. 
    4) The crucial point. Now either the two clicks are within an acceptance time,
    or they are not:
        a) If they are not, then remove the min from its list but do not update the loop index.
        This is to account for the case where we have one click in the first chain, and while 
        we wait for the second chain to be ready another click arrives in the first one.
        We would not discard this, and could be that it's closer to a click in the second chain.
        Experimentally, I would say that this is valid, since we could still try to store
        in other modes, and just leave the full one alone.
        b) If it is, all good. Add the first timetag to a list, plus the difference between
        the two clicks, update t_mark with the max and the latency, and go to the next pair.
    
I think this should cover all cases, and be a suitable representation of the experiment.

"""


# Setup figure with 3 vertically stacked subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(6, 8))

# Lists to collect data across storage times
rate_list = []       # Heralding rate
cok_list = []        # Successful match percentage
cmark_list = []      # Memory-not-ready percentage
cout_list = []       # Timeout percentage
tmed_list = []       # Median delay
tav_list = []        # Average delay
del_list = []        # List of all delays (for histogram)


# Loop over storage times (t_out = tau_AFC + tau_SW)
for to in t_out:
    # Run event correlation logic
    success, click, event, delay_list, counter_tout, counter_mark, counter_ok, mm = learn_statistics_either_var2(
        Ctrigger, Csignal, to, lat
    )

    # Total measurement duration for this setting
    tmeasure = success[-1] - success[0]
    rate = len(success) / tmeasure
    rate_list.append(rate)

    # Delay stats
    med = np.median(sorted(delay_list))
    av = np.mean(delay_list)
    tmed_list.append(med)
    tav_list.append(av)

    # Counter stats (as fractions)
    ctot = counter_tout + counter_ok + counter_mark
    cok_list.append(counter_ok / ctot)
    cmark_list.append(counter_mark / ctot)
    cout_list.append(counter_tout / ctot)

    # Save delay values for histogram
    del_list.append(sorted(delay_list))


# -------------------------------
# Plotting results
# -------------------------------

# Heralding rate vs storage time
ax1.plot(t_out * 1e3, rate_list)

# Counters vs storage time
ax3.plot(t_out * 1e3, cok_list, label='Matched Events')
ax3.plot(t_out * 1e3, cmark_list, label='Too Early (Not Ready)')
ax3.plot(t_out * 1e3, cout_list, label='Too Late (Timeout)')

# Normalized expected heralding limit for reference
norm = rate / ((2/3) * (Rid * eta_duty_mem * trial_frac))

# Histogram of delay times from last iteration
delay_list_hist = np.histogram(delay_list, bins=200)
norm_2 = 1.3 * max(delay_list_hist[0])
binsize = delay_list_hist[1][1] - delay_list_hist[1][0]

# Delay distribution (Parallel chain)
ax2.plot(1e3 * delay_list_hist[1][:-1], delay_list_hist[0], label='Parallel')

# Exponential decay fit overlay (expected shape)
ax2.plot(
    1e3 * delay_list_hist[1][:-1],
    norm_2 * np.exp(-(Rid * trial_frac) * delay_list_hist[1][:-1]),
    label='Exp Decay',
    c='k', ls='--'
)


# -------------------------------
# Plot Formatting
# -------------------------------

ax1.set(xlabel=r'$t_{out}$ (ms)', ylabel='Heralding Rate (counts/s)')
ax2.set(xlabel='Time Difference (ms)', ylabel='# Events')
ax3.set(xlabel=r'$t_{out}$ (ms)', ylabel='% of Events')

ax1.legend(loc='lower right', shadow=True)
ax2.legend(loc='upper right', shadow=True)
ax3.legend(loc='lower right', shadow=True)

ax1.set(title=f'1.5 MHz/mW & {P} mW - L = {L} km @ {alpha} dB/km')

plt.tight_layout()
plt.show()



# This is mostly for debugging

# fig, ax = plt.subplots()
# ax.plot(t_out*1e3, np.exp(-2*(tau_SW/T_eff)**M_m))
# # ax.bar(1e03*delay_list_hist[1][:-1], delay_list_hist[0], width = binsize*1e03)
# # ax.plot(event)
# # ax.scatter(range(len(Ctrigger)), Ctrigger)
# # ax.scatter(range(len(click)), click)
# # ax.plot(delay_list, color = back_col)
# # ax.plot(np.array(success[1:]) - np.array(success[:-1]))
# # ax.plot(np.array(click[1:]) - np.array(click[:-1]))
# # ax.plot(np.array(success) - np.array(click))
# # ax.bar(count_hist[1][:-1], count_hist[0])
# ax.set(xlabel = r'Delay (ms)', ylabel = 'Events')
# plt.tight_layout()
# plt.show()


#%%
"""
CALCULATIONS

Final rate, after mapping back to light and into polarisation. Here what you need
is to multiply the cumulative heralding rate and the decaying efficiency, and to sum
the result. I think it's something like the integral of the derivative...

"""

fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(6, 8))


# ----------------------------------
# 1. Heralding Rate (already computed)
# ----------------------------------
RH = np.array(rate_list)     # Heralding rate vs t_out
R_lim = max(RH)              # Max heralding rate for reference

ax1.plot(1e3 * t_out, RH, c=purple_2)


# ----------------------------------
# 2. Quantum Memory Efficiency Terms
# ----------------------------------

# Effective spin-wave storage efficiency (decays with SW time)
eta_coh = np.exp(-2 * (tau_SW / T_eff) ** M_m)

# AFC storage efficiency (with exponential decay due to T2)
eta_AFC = eta_AFC0 * np.exp(-4 * tau_AFC / T2)

# Total QM efficiency = AFC + control pulse loss
eta_QM = eta_AFC * eta_CP ** 2

# Signal transmission through memory
eta_QN = etaH * eta_T1 * eta_QM

# Final mapping back to photon (detector, filter, fiber)
eta_map = etaDs * eta_T3 * eta_T2


# ----------------------------------
# 3. Coincidence Rate
# ----------------------------------

# Accumulated coincidence rate, considering:
# - QM decay
# - Delta in RH between points
# - Time-dependent efficiency
Rcoinc = np.array(list(accumulate(
    0.5 * eta_QN ** 2 * eta_map ** 2 *
    np.exp(-2 * ((tau_min + t_comm) / T_eff) ** M_m) *
    (RH - np.insert(RH[:-1], 0, 0)) *
    eta_coh
)))

ax2.plot(1e3 * t_out, Rcoinc, c=yellow)
ax2.set(ylabel='Coincidence Rate (counts/s)')


# ----------------------------------
# 4. Fidelity Calculation
# ----------------------------------

# Median time-difference-based coherence decay
eta_coh_med = np.exp(-2 * (np.array(tmed_list) / T_eff) ** M_m)

# g² for spin-wave memory output
g2sw = (
    eta_QN * eta_coh_med * eta_map /
    ((etaH * eta_T1 * eta_AFC * eta_map) / g2 + NF)
) + 1

# Diagonal terms of output density matrix
p10 = 0.5 * eta_QN
p01 = p10
p11 = 4 * p10 * p01 / g2sw ** 2 * (g2sw - 1)

# Interference visibility
V = Vphase * etaOv * (g2sw - 1) / (g2sw + 1)

# Effective fidelity of entangled state
Feff = 0.5 * (1 + V) * (p10 + p01) / (p10 + p01 + p11)
F = Feff ** 2

ax3.plot(1e3 * t_out, F * 100, c=cmap(2))
ax3.set(ylim=(0, 100), ylabel='Fidelity (%)')


# ----------------------------------
# Plot Enhancements & Theoretical References
# ----------------------------------

# Plot expected heralding limit and decay
ax1.plot(
    t_out * 1e3,
    R_lim * (1 - np.exp(-2 * Rid * trial_frac * t_out)),
    label=r'$1 - e^{-2 R_{\mathrm{sent}} t_{\mathrm{frac}} t}$',
    c=purple_d, ls='--'
)
ax1.axhline(
    2 * Rid / 3 * eta_duty_mem * trial_frac,
    c=purple_2, ls='-.',
    label=r'$2R_{\mathrm{det}}/3$'
)

# Show quantum memory efficiency as second axis
ax2par = ax2.twinx()
ax2par.plot(
    1e3 * t_out,
    eta_QM * np.exp(-2 * (tau_SW / T_eff) ** M_m),
    c=brown, ls='-.',
    label='QM Efficiency'
)
ax2par.set(ylabel='Efficiency (%)')

# Show g² on second y-axis
ax3.plot(1e3 * t_out, F ** 2 * 100, c=cmap(2), ls='dotted', label=r'$F^2$')
ax3.axhline(50, c=cmap(3), ls='-.', label='Teleportation Limit')

ax3par = ax3.twinx()
ax3par.plot(1e3 * t_out, g2sw, c=cmap(1), ls='-.', label=r'$g^{(2)}$')
ax3par.set(ylabel=r'$g^{(2)}$')

# Labels & titles
ax1.set(
    ylabel='Heralding Rate (counts/s)',
    title=f"{bright / M * 1e-3:.0f} kHz/mW/mode, {P} mW, {eta_duty:.2f} duty - {L} km, {alpha} dB/km"
)
ax2.set(
    title=rf"$\eta_{{AFC}}$ = {eta_AFC0}, $\eta_{{CP}}$ = {eta_CP}, $\eta_H$ = {etaH}, "
          rf"$t_{{frac}}$ = {trial_frac:.1f}, $T_{{eff}}$ = {T_eff * 1e3:.0f} ms"
)
ax3.set(
        xlabel='Storage Time (ms)', 
        title=rf"{time_span/dt:.0f} modes, Tot. Trans = {eta_T1 * eta_T2 * eta_T3:.2f}, Measured V = {Vphase * etaOv:.2f}"
)

# Legends
ax1.legend(loc='lower right', shadow=True)
ax2par.legend(loc='lower right', shadow=True)
ax3.legend(loc='upper right', shadow=True)
ax3par.legend(loc='lower right', shadow=True)

plt.xlabel('Maximum storage time (ms)')
plt.tight_layout()
plt.show()


#%%
"""
QIA PROTOTYPE

Here I am extrapolating for the final rate, considering a Jiang-style repeater where
we read out all four memories in the central node. I just take the mathematical
behaviour of rates in the individual link and extend it to now two links.
"""

# Target heralding rate for a 500 km repeater
RH4_0 = 2 * Rid / 3 * eta_duty_mem * trial_frac

# Expected time-dependent heralding rate - 90% arbitrary parameter from above
RH4_time = 0.9 * (RH4_0 / 2) * (1 - np.exp(-2 * RH4_0 * t_out))

# Accumulated coincident rate over time
RH4_her = np.array(list(accumulate(
    0.5 * eta_QN ** 2 * eta_map ** 2 *
    np.exp(-2 * ((tau_min + t_comm) / T_eff) ** M_m) *
    (RH4_time - np.insert(RH4_time[:-1], 0, 0)) *
    eta_coh
)))

# Plot QIA-style heralding rate and fidelity
fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

# Heralding rate
ax1.plot(1e3 * t_out, RH4_her, c=purple_d, ls='--')
ax1.set(
    ylabel='Repeater Heralding Rate',
    title=rf'{4 * L:.0f} km Repeater Heralding Rate (Jiang-style)'
)

# Accumulated coincidence rate, considering:
# - QM decay
# - Delta in RH between points
# - Time-dependent efficiency
Rcoinc = np.array(list(accumulate(
    0.5 * eta_QN ** 2 * eta_map ** 2 *
    np.exp(-2 * ((tau_min + t_comm) / T_eff) ** M_m) *
    (RH4_her - np.insert(RH4_her[:-1], 0, 0)) *
    eta_coh
)))

ax2.plot(1e3 * t_out, Rcoinc, c=yellow)
ax2.set(ylabel='Coincidence Rate (counts/s)')

# Fidelity
ax3.plot(1e3 * t_out, F ** 2 * 100, c=cmap(2), label='F²')
ax3.axhline(50, c=cmap(1), ls='-.', label='Teleportation Limit')
ax3.set(
    ylim=(0, 100),
    xlabel='Max Storage Time (ms)',
    ylabel='Fidelity (%)'
)

ax3.legend(loc='lower right', shadow=True)
plt.tight_layout()
plt.show()


#%%
"""
QIA SGA2 WELINQ conparison
"""

RH_us = np.array([ 2.5984078 ,  5.52797712,  6.23181042,  6.23181042,  6.69900415,
        8.65690241, 11.77851127, 13.62833534, 14.44608691, 14.44608691,
       14.60231363, 15.64062153, 17.61317624, 19.12528933, 19.7761313 ,
       19.82638914, 19.90198177, 20.59846192, 22.06670414, 23.33846101,
       23.86966902, 23.9763002 , 23.98278554, 24.39981752, 25.37859487,
       26.48157337, 27.04026692, 27.17200736, 27.17200736, 27.34489392,
       27.93378383, 28.59496451, 29.04536684, 29.17854825, 29.17854825,
       29.26562409, 29.73911572, 30.44680294, 30.86768695, 31.02056277,
       31.02056277, 31.04798763, 31.3154144 , 31.81616181, 32.19502134,
       32.39959098, 32.40256451, 32.41048913, 32.72989812, 33.09877899,
       33.54809918, 33.76197904, 33.78557279, 33.7943673 , 33.93264426,
       34.11603569, 34.52291306, 34.73958506, 34.76883186, 34.76883186,
       34.87933086, 35.01620516, 35.3666478 , 35.52524154, 35.57255799,
       35.57255799, 35.6012628 , 35.78086288, 36.01729173, 36.19534606,
       36.27021357, 36.27021357, 36.28496324, 36.41635163, 36.53112352,
       36.70372898, 36.80034045, 36.80034045, 36.80884199, 36.8996088 ,
       37.10945904, 37.229471  , 37.29956393, 37.29956393, 37.32763735,
       37.40170594, 37.55472617, 37.73464165, 37.80152022, 37.80862158,
       37.80862158, 37.84399517, 37.95895079, 38.10311226, 38.15775215,
       38.20571312, 38.20571312, 38.21370535, 38.32582276, 38.46710386])
RH_lim_us = 41.931

Rcoinc_us = np.array([0.01965713, 0.04178963, 0.0470906 , 0.0470906 , 0.05056695,
       0.06500114, 0.08775062, 0.10104585, 0.10682829, 0.10682829,
       0.10788961, 0.11477688, 0.1275189 , 0.13700573, 0.14096086,
       0.14125587, 0.14168329, 0.14546598, 0.15310365, 0.15942169,
       0.16193468, 0.16241361, 0.16244119, 0.16411457, 0.16780995,
       0.17171621, 0.17356654, 0.17397329, 0.17397329, 0.17443285,
       0.17587842, 0.17737246, 0.17830632, 0.17855888, 0.17855888,
       0.17869566, 0.17936928, 0.18027808, 0.18076435, 0.18092272,
       0.18092272, 0.18094533, 0.18114105, 0.18146523, 0.18168145,
       0.18178401, 0.18178532, 0.18178835, 0.1818947 , 0.1820011 ,
       0.182113  , 0.18215882, 0.18216315, 0.18216453, 0.18218299,
       0.18220376, 0.18224269, 0.18226014, 0.18226212, 0.18226212,
       0.18226732, 0.18227266, 0.18228397, 0.18228819, 0.18228922,
       0.18228922, 0.18228964, 0.18229175, 0.18229401, 0.18229538,
       0.18229584, 0.18229584, 0.1822959 , 0.1822963 , 0.18229659,
       0.18229692, 0.18229707, 0.18229707, 0.18229708, 0.18229714,
       0.18229726, 0.18229731, 0.18229733, 0.18229733, 0.18229734,
       0.18229735, 0.18229736, 0.18229738, 0.18229738, 0.18229738,
       0.18229738, 0.18229738, 0.18229739, 0.18229739, 0.18229739,
       0.18229739, 0.18229739, 0.18229739, 0.18229739, 0.18229739])

RH_w = np.array([0.06281664, 0.06281664, 0.06281664, 0.06281664, 0.06281664,
       0.06281664, 0.1833825 , 0.1833825 , 0.1833825 , 0.1833825 ,
       0.1833825 , 0.1833825 , 0.28526166, 0.28933683, 0.28933683,
       0.28933683, 0.28933683, 0.28933683, 0.32601333, 0.38866083,
       0.38866083, 0.38866083, 0.38866083, 0.38866083, 0.38866083,
       0.48689477, 0.48689477, 0.48689477, 0.48689477, 0.48689477,
       0.48689477, 0.61505695, 0.61505695, 0.61505695, 0.61505695,
       0.61505695, 0.61505695, 0.73952922, 0.73952922, 0.73952922,
       0.73952922, 0.73952922, 0.73952922, 0.83295616, 0.83295616,
       0.83295616, 0.83295616, 0.83295616, 0.83295616, 0.85972916,
       0.90643854, 0.90643854, 0.90643854, 0.90643854, 0.90643854,
       0.90643854, 0.99728582, 0.99728582, 0.99728582, 0.99728582,
       0.99728582, 0.99728582, 1.07543135, 1.07543135, 1.07543135,
       1.07543135, 1.07543135, 1.07543135, 1.14158074, 1.14158074,
       1.14158074, 1.14158074, 1.14158074, 1.14158074, 1.2081753 ,
       1.2081753 , 1.2081753 , 1.2081753 , 1.2081753 , 1.2081753 ,
       1.25149122, 1.27367856, 1.27367856, 1.27367856, 1.27367856,
       1.27367856, 1.27367856, 1.35045145, 1.35045145, 1.35045145,
       1.35045145, 1.35045145, 1.35045145, 1.41614081, 1.41614081,
       1.41614081, 1.41614081, 1.41614081, 1.41614081, 1.47317286])
RH_lim_w = 4.193

Rcoinc_w = np.array([0.00047521, 0.00047521, 0.00047521, 0.00047521, 0.00047521,
       0.00047521, 0.00135386, 0.00135386, 0.00135386, 0.00135386,
       0.00135386, 0.00135386, 0.00201197, 0.00203754, 0.00203754,
       0.00203754, 0.00203754, 0.00203754, 0.00222832, 0.00253956,
       0.00253956, 0.00253956, 0.00253956, 0.00253956, 0.00253956,
       0.00288746, 0.00288746, 0.00288746, 0.00288746, 0.00288746,
       0.00288746, 0.00317706, 0.00317706, 0.00317706, 0.00317706,
       0.00317706, 0.00317706, 0.0033369 , 0.0033369 , 0.0033369 ,
       0.0033369 , 0.0033369 , 0.0033369 , 0.00339739, 0.00339739,
       0.00339739, 0.00339739, 0.00339739, 0.00339739, 0.00340511,
       0.00341674, 0.00341674, 0.00341674, 0.00341674, 0.00341674,
       0.00341674, 0.00342543, 0.00342543, 0.00342543, 0.00342543,
       0.00342543, 0.00342543, 0.00342796, 0.00342796, 0.00342796,
       0.00342796, 0.00342796, 0.00342796, 0.00342859, 0.00342859,
       0.00342859, 0.00342859, 0.00342859, 0.00342859, 0.00342875,
       0.00342875, 0.00342875, 0.00342875, 0.00342875, 0.00342875,
       0.00342878, 0.00342878, 0.00342878, 0.00342878, 0.00342878,
       0.00342878, 0.00342878, 0.00342879, 0.00342879, 0.00342879,
       0.00342879, 0.00342879, 0.00342879, 0.00342879, 0.00342879,
       0.00342879, 0.00342879, 0.00342879, 0.00342879, 0.00342879])


# fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(6, 5))
# ax1.plot(1e3 * t_out, RH_us)
# ax1.plot(1e3 * t_out, RH_w)

# ax2.plot(1e3 * t_out, Rcoinc_us)
# ax2.plot(1e3 * t_out, Rcoinc_w)

# plt.xlabel('Maximum storage time (ms)')
# plt.tight_layout()
# plt.show()


# fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(6, 5))
# ax1.plot(1e3 * t_out, RH_us, c=purple_2)
# ax1x = ax1.twinx()
# ax1x.plot(1e3 * t_out, RH_w, c=purple_d, ls='--')

# ax2.plot(1e3 * t_out, Rcoinc_us, c=yellow)
# ax2x = ax2.twinx()
# ax2x.plot(1e3 * t_out, Rcoinc_w, c=brown, ls='--')

# plt.xlabel('Maximum storage time (ms)')
# plt.tight_layout()
# plt.show()


# fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(6, 5))
# ax1.plot(1e3 * t_out, RH_us, c=purple_2)
# ax1x = ax1.twinx()
# ax1x.plot(1e3 * t_out, Rcoinc_us, c=yellow, ls='--')

# ax2.plot(1e3 * t_out, RH_w, c=purple_d)
# ax2x = ax2.twinx()
# ax2x.plot(1e3 * t_out, Rcoinc_w, c=brown, ls='--')

# plt.xlabel('Maximum storage time (ms)')
# plt.tight_layout()
# plt.show()



#%%

# fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(6, 5))
# ax1.plot(1e3 * t_out, RH_us, c=purple_2)
# ax1.axhline(
#     RH_lim_us,
#     c=purple_d, ls='-.',
#     label=r'$2R_{\mathrm{det}}/3$'
# )

# ax2.plot(1e3 * t_out, Rcoinc_us, c=yellow)
# ax2par = ax2.twinx()
# ax2par.plot(
#     1e3 * t_out,
#     eta_QM * np.exp(-2 * (tau_SW / T_eff) ** M_m),
#     c=brown, ls='-.',
#     label='QM Efficiency'
# )
# ax2par.set(ylabel='Efficiency (%)')

# plt.xlabel('Maximum storage time (ms)')
# plt.tight_layout()
# plt.show()


# fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(6, 5))
# ax1.set_yticks([])
# ax1par = ax1.twinx()
# ax1par.plot(1e3 * t_out, RH_w, c=purple_2)
# ax1par.axhline(
#     RH_lim_w,
#     c=purple_d, ls='-.',
#     label=r'$2R_{\mathrm{det}}/3$'
# )

# ax2.plot(
#     1e3 * t_out,
#     eta_QM * np.exp(-2 * (tau_SW / T_eff) ** M_m),
#     c=brown, ls='-.',
#     label='QM Efficiency'
# )
# ax2par = ax2.twinx()
# ax2par.plot(1e3 * t_out, Rcoinc_w, c=yellow)
# ax2par.set(ylabel='Efficiency (%)')

# plt.xlabel('Maximum storage time (ms)')
# plt.tight_layout()
# plt.show()