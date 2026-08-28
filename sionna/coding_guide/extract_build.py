"""Build the code-along file for the coding guide.

Extracts the two-way synchronization implementation VERBATIM from
ota_sync/core.py, ota_sync/sdr.py, and ota_sync/coherent.py into one
self-contained file, build_twoway.py, in pedagogical order. Section
markers (# %% ...) drive the slide generator, so the slides' code
snippets are, by construction, exactly the code in the file -- and
the file is, by construction, exactly the repository's code.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "ota_sync"

HEADER = '''"""Two-way over-the-air phase synchronization, built from the ground up.

Every line below is verbatim from ota_sync/ (core.py, sdr.py,
coherent.py) -- only the import header differs, because everything
lives in this one file. Typing this file section by section, as the
coding guide walks it, produces the repository's actual two-way
implementation; running it reproduces the repository's numbers
exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
from sionna.phy import config as sionna_config
from sionna.phy.channel import AWGN, ApplyTimeChannel
from sionna.phy.channel.tr38901 import TDL
from sionna.phy.channel.utils import (
    cir_to_time_channel,
    time_lag_discrete_time_channel,
)

REAL_DTYPE = torch.float64
COMPLEX_DTYPE = torch.complex128
'''

FOOTER = '''

# %% SECTION: Run it
# %% NOTE: The finished program. Two clocks starting 1500\\,Hz and
# %% NOTE: 1.2\\,rad apart, a 3GPP multipath channel, the full
# %% NOTE: hardware impairment chain -- and the loop holds them
# %% NOTE: aligned to a few degrees. Because every line above is the
# %% NOTE: repository's own code, these numbers are identical to
# %% NOTE: \\texttt{ota\\_sync.run\\_two\\_way\\_simulation}'s output,
# %% NOTE: bit for bit.
if __name__ == "__main__":
    result = run_two_way_simulation(SDRSimulationConfig())
    print(f"detection rate     : {result.detection_rate:.2f}")
    print(f"steady phase rms   : {result.steady_state_phase_rms:.4f} rad")
    print(f"mean coherent gain : {result.mean_coherent_gain * 100:.1f} %")
    print(f"airtime fraction   : {result.airtime_fraction * 100:.2f} %")
    print(f"final freq error   : {result.final_frequency_error_hz:.3f} Hz")
'''

# (file, first_line, last_line, section_title, note_lines)
PARTS = [
    ("core.py", 24, 27, "wrap\\_phase", [
        "A phase is an angle: $370^\\circ$ \\emph{is} $10^\\circ$. Every",
        "phase subtraction in this program goes through this one-liner,",
        "or differences near the boundary come out wrong by a full",
        "turn. It appears on nearly every line that touches a phase;",
        "when debugging this kind of code, suspect a missing wrap",
        "first. (Repository: \\texttt{core.py:24}.)",
    ]),
    ("core.py", 50, 62, "Two small linear-algebra helpers", [
        "\\texttt{\\_as\\_matrix} is convenience. \\texttt{\\_covariance\\_root}",
        "matters: to draw correlated noise with covariance $Q$ you need",
        "a matrix square root of $Q$; an eigendecomposition gives one",
        "that tolerates positive-\\emph{semi}definite $Q$ (a zero",
        "eigenvalue = a noiseless direction), which a plain Cholesky",
        "factorization would reject. (\\texttt{core.py:50--62}.)",
    ]),
    ("core.py", 133, 141, "resolve\\_device", [
        "Runs on the graphics card when one is available, the",
        "processor otherwise. (\\texttt{core.py:133}.)",
    ]),
    ("core.py", 175, 225, "The Oscillator -- the enemy", [
        "The clock model: state $[\\theta, \\omega]$ (phase, frequency).",
        "The transition matrix says phase advances by frequency",
        "$\\times$ time. The noise is a \\textbf{random walk in both",
        "components} -- drawn through the covariance root so the two",
        "components are correctly correlated -- and random walks",
        "accumulate; they never average away. \\texttt{apply\\_correction}",
        "is the actuator the loop will drive; nothing else may touch",
        "the state. Keep the transition matrix in mind: the Kalman",
        "filter carries an exact copy of it, so filter and physics",
        "agree by construction. (\\texttt{core.py:175--225}.)",
    ]),
    ("sdr.py", 33, 94, "SDRSimulationConfig -- every physical assumption", [
        "One frozen dataclass holds every knob: timing (50\\,ms",
        "intervals, 1\\,MHz sampling), the channel (3GPP model ``D'',",
        "100\\,ns delay spread, 915\\,MHz), the pilot structure",
        "($16\\times16$ short, $2047\\times2$ long), the oscillator noise",
        "levels, every hardware impairment, and the loop's actuation",
        "limits (16-bit phase corrections, one-interval command",
        "latency, 1\\,ms turnaround). Every experiment in the",
        "repository is a variation of these fields -- never edited",
        "code. (\\texttt{sdr.py:33--94}.)",
    ]),
    ("sdr.py", 96, 157, "Config validation", [
        "The dataclass rejects nonsense at construction: negative",
        "noise levels, a cyclic prefix longer than its sequence, too",
        "few training repeats. Fail loudly at setup, not silently",
        "mid-run. The \\texttt{sample\\_period} property is the only",
        "derived quantity. (\\texttt{sdr.py:96--157}.)",
    ]),
    ("sdr.py", 160, 191, "Three small data carriers", [
        "\\texttt{SyncPreamble}: the pilot waveform plus the layout",
        "facts the receiver needs. \\texttt{IQCapture}: what one",
        "capture returns -- the impaired samples, an impairment-free",
        "\\emph{oracle} copy (ground truth for diagnostics), and the",
        "end value of the intra-frame oscillator walk, which the loop",
        "must carry into the clock state. \\texttt{SDRMeasurement}:",
        "what the receiver reports -- detected?, phase, frequency,",
        "timing, and the detection metric. (\\texttt{sdr.py:160--191}.)",
    ]),
    ("sdr.py", 253, 288, "The pilot: Zadoff--Chu training fields", [
        "Constant-amplitude chirps with a single-spike",
        "autocorrelation -- the same family LTE and 5G use. The",
        "preamble is $16\\times16$ short repeats (for coarse timing and",
        "frequency) then $2047\\times2$ long sequences with a cyclic",
        "prefix (for fine frequency and phase). Why 2047: the",
        "frequency-estimate error times the 50\\,ms interval is phase",
        "drift the loop must ride out, and it must stay far inside",
        "the $\\pm\\pi/2$ branch boundary you will meet later.",
        "(\\texttt{sdr.py:253--288}.)",
    ]),
    ("sdr.py", 290, 325, "Converter and clock imperfections", [
        "\\texttt{\\_quantize\\_iq}: a $b$-bit converter -- clip to full",
        "scale, round to the grid, report the clip rate.",
        "\\texttt{\\_soft\\_limit}: the power amplifier's gentle",
        "saturation. \\texttt{\\_resample\\_clock\\_offset}: the receiver's",
        "sample clock runs at a slightly different rate than the",
        "transmitter's, so the samples are linearly re-interpolated at",
        "the offset rate plus a fractional delay.",
        "(\\texttt{sdr.py:290--325}.)",
    ]),
    ("sdr.py", 328, 391, "Flicker frequency noise", [
        "Real oscillators have $1/f$ frequency noise on top of the",
        "white walks. The standard time-domain surrogate: a bank of",
        "four first-order decaying processes with log-spaced time",
        "constants, summed. The class also exposes its per-interval",
        "\\texttt{innovation\\_variance} -- the loop adds it to the",
        "filter's process noise, so the filter knows about this noise",
        "source too. (\\texttt{sdr.py:328--391}.)",
    ]),
    ("sdr.py", 394, 473, "SDRRadioLink: the channel", [
        "The propagation channel is Sionna's 3GPP tapped-delay-line,",
        "converted to a 16-tap discrete-time filter, one realization",
        "per sync interval. \\textbf{\\texttt{mirror\\_of} is reciprocity",
        "in one argument}: the reverse link of a two-way pair shares",
        "the forward link's taps and shadowing -- the physical fact",
        "the entire method stands on. The noise floor is fixed at the",
        "nominal channel gain, so fading changes the actual",
        "signal-to-noise ratio, as in the field.",
        "(\\texttt{sdr.py:394--473}.)",
    ]),
    ("sdr.py", 475, 508, "Link helpers: jitter and shadowing", [
        "\\texttt{\\_random\\_start}: the capture window lands with",
        "random timing jitter. \\texttt{\\_channel\\_for\\_frame}: this",
        "interval's taps, shaped for Sionna's channel operator.",
        "\\texttt{\\_step\\_shadowing}: slow log-normal fading as a",
        "correlated process -- and the mirrored link reuses the",
        "forward link's draw (reciprocity again).",
        "(\\texttt{sdr.py:475--508}.)",
    ]),
    ("sdr.py", 510, 544, "capture(), transmit side", [
        "One direction of one exchange, in physical order: the",
        "receiver's sample-clock error moves the arrival window",
        "(whole samples step it; the sub-sample residue stays as a",
        "fractional delay); the transmitter imprints \\emph{its}",
        "oscillator on the waveform; the power amplifier soft-clips;",
        "the DAC quantizes; Sionna's channel convolves; shadowing",
        "scales. (\\texttt{sdr.py:510--544}.)",
    ]),
    ("sdr.py", 545, 618, "capture(), receive side", [
        "Thermal noise at the fixed floor; sample-clock resampling;",
        "down-conversion with the \\emph{receiver's} oscillator (what",
        "remains is the clock difference plus the channel); the",
        "oscillator's own random walk \\emph{during} the frame -- whose",
        "end value is returned so the loop keeps the noise continuous",
        "across frames; white phase jitter; IQ imbalance; automatic",
        "gain control; DC offset; 12-bit quantization. Every phase",
        "error the filter must fight is created here, physically.",
        "(\\texttt{sdr.py:545--618}.)",
    ]),
    ("sdr.py", 621, 659, "The receiver, stage 1: coarse timing and frequency", [
        "The receiver knows neither when the pilot arrives nor the",
        "frequency offset. Correlating the signal with itself 16",
        "samples later solves both at once: when the repeating short",
        "field fills the window, consecutive repeats match -- the",
        "normalized magnitude peaks (timing), and the angle between",
        "repeats is the offset accumulated over 16 samples",
        "(frequency, unambiguous to $\\pm f_s/32 = \\pm31$\\,kHz).",
        "(\\texttt{sdr.py:621--659}.)",
    ]),
    ("sdr.py", 661, 754, "The receiver, stages 2--3: detect, refine, read the phase", [
        "Stage 2: after coarse derotation, the \\emph{whole} preamble",
        "is matched against every window; the normalized peak gives",
        "precise timing, and its height is the detection metric --",
        "below the 0.25 threshold the capture is declared failed and",
        "the loop must coast. Stage 3: the two long fields are",
        "identical and 2175 samples apart; the angle between them",
        "refines the frequency to $\\sim$0.1\\,Hz. Finally, derotate",
        "with the refined frequency (indices centered so the phase is",
        "referenced to frame \\emph{center}) and matched-filter the",
        "long fields: the correlation angle is the carrier phase --",
        "clock offset \\textbf{plus the channel's phase}. Separating",
        "those two is the loop's job. (\\texttt{sdr.py:661--754}.)",
    ]),
    ("sdr.py", 757, 815, "R from physics, and quantized corrections", [
        "The filter's measurement covariance is \\textbf{derived, not",
        "tuned}: the textbook estimation bound for a pilot of this",
        "length and signal-to-noise ratio, plus the phase noise the",
        "oscillator smears across the frame, plus per-sample jitter.",
        "This is why no gain is hand-tuned anywhere.",
        "\\texttt{\\_quantize\\_correction} snaps commands to what",
        "hardware can apply: 16-bit phase steps, 0.01\\,Hz frequency",
        "steps. (\\texttt{sdr.py:757--815}.)",
    ]),
    ("core.py", 286, 339, "The extended Kalman filter: state, predict, and the circle", [
        "The filter tracks the \\emph{relative} state",
        "$[\\hat\\theta, \\hat\\omega]$ with the same transition matrix as",
        "the Oscillator. \\texttt{predict}: push the estimate through",
        "the drift model, inflate the covariance -- ``the clocks kept",
        "drifting while I wasn't looking.'' The measurement is a",
        "\\textbf{point on the unit circle}, $[\\cos\\phi, \\sin\\phi,",
        "\\omega]$, so no innovation ever wraps; the price is a",
        "nonlinear $h$, linearized by the Jacobian -- which is the",
        "entire meaning of ``extended.'' (\\texttt{core.py:286--339}.)",
    ]),
    ("core.py", 341, 382, "The iterated update, Joseph form, and the reset", [
        "Three deliberate choices. \\textbf{Iterate}: with a large",
        "offset one linearized shot under-corrects (the comment says",
        "it best: $\\sin(-1.4)$ is not $\\approx -1.4$) -- so update,",
        "re-linearize, repeat until converged. \\textbf{Joseph form}:",
        "the textbook covariance shortcut can go asymmetric or",
        "negative under rounding; this form cannot.",
        "\\textbf{reset\\_after\\_correction}: after commanding the",
        "clock to shift, subtract the command from the estimate --",
        "the filter tracks \\emph{what remains}. Skip it and every",
        "correction applies twice. (\\texttt{core.py:341--382}.)",
    ]),
    ("coherent.py", 42, 101, "The result object", [
        "Everything the run records, with the honest masks built in:",
        "\\texttt{steady\\_state\\_phase\\_rms} and",
        "\\texttt{mean\\_coherent\\_gain} only count intervals where",
        "\\texttt{detected \\& correction\\_active \\& calibrated} --",
        "steady state means \\emph{after} acquisition, actuation, and",
        "the $\\pi$ check, and the code says so explicitly.",
        "(\\texttt{coherent.py:42--101}.)",
    ]),
    ("coherent.py", 104, 119, "\\_pick\\_half\\_phase: the two candidate answers", [
        "The two-way trick divides a phase difference by two, and",
        "division by two on a circle loses one bit: the true offset",
        "is either the measurement or the measurement $+\\pi$ --",
        "\\emph{exactly} two candidates, at any signal-to-noise ratio.",
        "Once locked, the offset cannot physically jump by $\\pi$ in",
        "one interval, so the filter's own prediction picks the",
        "branch. At first contact there is no prediction -- that hole",
        "is plugged by the one-time power check inside the loop.",
        "(\\texttt{coherent.py:104--119}.)",
    ]),
    ("coherent.py", 122, 231, "run\\_two\\_way\\_simulation: setup", [
        "Seeds (note \\texttt{sionna\\_config.seed} -- the channel is",
        "pinned by Sionna's own seed, not torch's; a real trap), two",
        "oscillators sharing one covariance, the reverse link as a",
        "\\texttt{mirror\\_of} the forward link, and the filter's",
        "matrices assembled from physics: $R$ = the derived",
        "covariance \\emph{halved} (the half-difference averages two",
        "independent captures); $Q$ = \\emph{twice} the per-oscillator",
        "covariance (two independent clocks make the relative state)",
        "+ the white-FM walk over one interval + the flicker",
        "innovation. Initial covariance: phase $\\pm\\pi$, frequency",
        "$\\pm50$\\,kHz -- honest ignorance. Airtime is accounted",
        "exactly: two captures per interval.",
        "(\\texttt{coherent.py:122--231}.)",
    ]),
    ("coherent.py", 233, 264, "The loop: drift, delayed corrections, the pi check", [
        "Each interval starts with physics: both clocks walk; the",
        "master absorbs last frame's carried oscillator walk and the",
        "flicker step. A correction scheduled",
        "\\texttt{correction\\_latency\\_intervals} ago lands now. Then",
        "the \\textbf{one-time $\\pi$ calibration}: the acquisition",
        "branch was a guess (wrong about half the time, with the pair",
        "transmitting in opposite phase while every signal reads",
        "``locked''); three settled corrections after lock, one",
        "coarse combined-power check -- destructive $\\Rightarrow$",
        "flip by $\\pi$, once. The comment explains why the filter",
        "needs no reset: the measurement cannot see a $\\pi$ shift.",
        "(\\texttt{coherent.py:233--264}.)",
    ]),
    ("coherent.py", 266, 318, "The loop: two captures and the gap between them", [
        "Bookkeeping first: the correction-free ``physical'' clock",
        "(what hardware would do with no synchronization at all)",
        "yields the sample-clock offset for the captures. Then the",
        "forward capture; its intra-frame oscillator walk joins the",
        "true state \\emph{before} the reverse frame fires -- one",
        "continuous noise process. Between directions sits a real",
        "1\\,ms transmit/receive turnaround: both clocks advance at",
        "their current frequencies and pick up random walk. Then the",
        "reverse capture, and the leftover walk for the rest of the",
        "interval is drawn and carried to the next iteration.",
        "(\\texttt{coherent.py:266--318}.)",
    ]),
    ("coherent.py", 320, 363, "The loop: the half-difference, acquire, track", [
        "The central lines of the method. Forward measures",
        "$\\theta_m-\\theta_s+\\varphi_{ch}$; reverse measures",
        "$\\theta_s-\\theta_m+\\varphi_{ch}$; half the difference",
        "cancels the channel exactly (reciprocity). The turnaround",
        "drift is backed out with the loop's \\emph{own} frequency",
        "estimate. Then the filter: predict; tell it about a landed",
        "correction; on first detection seed the state from the",
        "measurement; afterwards pick the branch against the",
        "prediction and update on the circle. No detection",
        "$\\Rightarrow$ coast. (\\texttt{coherent.py:320--363}.)",
    ]),
    ("coherent.py", 364, 420, "The loop: actuate, record, and the figure of merit", [
        "The correction that will land $L$ intervals from now should",
        "cancel the offset \\emph{then} -- so the estimate is pushed",
        "$L$ steps through the transition matrix, quantized to what",
        "hardware can apply, and scheduled; the filter reset happens",
        "when it \\emph{lands}. The scoreboard is the true residual",
        "between the clocks -- something only a simulation can see --",
        "and the coherent gain it implies:",
        "$|1+e^{j\\,\\text{residual}}|^2/4 = \\cos^2(\\text{residual}/2)$.",
        "(\\texttt{coherent.py:364--420}.)",
    ]),
]


def main() -> None:
    sources = {
        name: (ROOT / name).read_text().splitlines()
        for name in ("core.py", "sdr.py", "coherent.py")
    }
    out = [HEADER]
    for fname, lo, hi, title, note in PARTS:
        out.append("\n")
        out.append(f"# %% SECTION: {title}\n")
        for line in note:
            out.append(f"# %% NOTE: {line}\n")
        body = sources[fname][lo - 1 : hi]
        out.append("\n".join(body) + "\n")
    out.append(FOOTER)
    Path(__file__).with_name("build_twoway.py").write_text("".join(out))
    total = sum(hi - lo + 1 for _, lo, hi, _, _ in PARTS)
    print(f"wrote build_twoway.py ({total} extracted lines, "
          f"{len(PARTS)} sections)")


if __name__ == "__main__":
    main()
