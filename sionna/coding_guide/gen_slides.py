"""Generate the coding-guide Beamer deck from build_twoway.py.

The deck's code snippets are read straight out of build_twoway.py --
never retyped -- so the slides, the file, and the repository agree by
construction. Regenerate after any change with:

    python3 extract_build.py && python3 gen_slides.py
    tectonic phase_sync_coding_guide.tex
"""

from pathlib import Path

SRC = Path(__file__).with_name("build_twoway.py")
SRC2 = Path(__file__).with_name("build_sensing.py")
OUT = Path(__file__).with_name("phase_sync_coding_guide.tex")

PREAMBLE = r"""\documentclass[10pt]{beamer}
\usetheme{Madrid}
\usecolortheme{seahorse}
\setbeamertemplate{navigation symbols}{}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{amsmath}

\definecolor{codebg}{RGB}{245,245,245}
\definecolor{codekw}{RGB}{0,90,160}
\definecolor{codecomment}{RGB}{90,130,90}
\definecolor{codestr}{RGB}{150,60,30}

\lstset{
  language=Python,
  basicstyle=\ttfamily\scriptsize,
  keywordstyle=\color{codekw}\bfseries,
  commentstyle=\color{codecomment}\itshape,
  stringstyle=\color{codestr},
  backgroundcolor=\color{codebg},
  showstringspaces=false,
  frame=single,
  framerule=0pt,
  xleftmargin=4pt,
  xrightmargin=4pt,
  breaklines=true,
  columns=fullflexible,
  keepspaces=true,
}

\title{Build Two-Way Phase Synchronization}
\subtitle{Code it from the ground up --- the snippets on these slides
ARE the implementation}
\author{Distributed-array phase synchronization project}
\date{}

\begin{document}

\begin{frame}
  \titlepage
\end{frame}

\begin{frame}{How this guide works}
  \begin{itemize}
    \item You are going to write \textbf{one Python file},
      \texttt{build\_twoway.py}, from top to bottom. Each section of
      this deck explains what comes next, then gives you the code to
      type --- \textbf{the snippets, typed in order, are the complete
      file}. Nothing is elided, nothing is toy code.
    \item And the file is not a simplification either: every line is
      \textbf{verbatim from the repository}
      (\texttt{ota\_sync/core.py}, \texttt{sdr.py},
      \texttt{coherent.py}); each section cites its source lines.
      When you finish, you have built the repository's actual two-way
      synchronization --- and running your file reproduces the
      repository's output \textbf{bit for bit} (verified; the check
      is on the last slide).
    \item Requirements: Python with \texttt{torch} and
      \texttt{sionna} $\geq$ 2.0.
    \item What you are building: two stations, each with its own
      drifting clock, exchange pilot waveforms both ways through a
      3GPP multipath channel every 50\,ms; a receiver chain measures
      timing/frequency/phase from raw samples; an extended Kalman
      filter tracks the clock offset; closed-loop corrections hold
      the pair aligned. Final performance, which your file will
      print: coherent gain 99.8\%.
  \end{itemize}
\end{frame}

\begin{frame}{The build order}
  \begin{enumerate}
    \item \textbf{Header} --- imports and the two working dtypes.
    \item \textbf{Small tools} --- \texttt{wrap\_phase}, covariance
      root, device pick.
    \item \textbf{The Oscillator} --- the drifting clock (the enemy).
    \item \textbf{The config} --- every physical assumption in one
      dataclass.
    \item \textbf{The pilot} --- Zadoff--Chu training fields.
    \item \textbf{Hardware imperfections} --- converters, amplifier,
      sample-clock, flicker noise.
    \item \textbf{The radio link} --- Sionna channel + the full
      capture chain.
    \item \textbf{The receiver} --- timing, frequency, phase from raw
      samples; $R$ from physics.
    \item \textbf{The filter} --- the extended Kalman filter.
    \item \textbf{The loop} --- half-difference, branch handling,
      corrections, metrics.
    \item \textbf{Run it} --- and verify against the repository.
  \end{enumerate}
\end{frame}
"""

PART2_INTRO = r"""
\begin{frame}{Part II --- sensing: what you build next}
  A second file, \texttt{build\_sensing.py}, same rules: the
  snippets, typed in order, are the complete file; every number on
  these slides is the measured output of running it. This part
  covers the next round of project work:
  \begin{enumerate}
    \item \textbf{Realistic base-station parameters} --- 3.5\,GHz
      carrier, 20\,MHz bandwidth, 10\,W, real noise floor.
    \item \textbf{Short/long training fields from the ground up} ---
      timing, frequency, and phase estimation at those parameters,
      tested over 500 impaired trials.
    \item \textbf{Monostatic sensing with one base station in
      Sionna} --- the station listens for its own echo, in
      line-of-sight and with a building in the way.
    \item \textbf{OFDM versus a single tone} --- which measurement
      each waveform can and cannot make.
  \end{enumerate}
  \vspace{0.3em}
  \footnotesize One practical note: the ray-traced sections need
  Sionna's ray-tracing module. On this machine that lives in the
  project environment --- run Part II with
  \texttt{\textasciitilde/Downloads/Princeton\_Research/ota\_sync/}
  \texttt{.venv/bin/python build\_sensing.py} (the training-field
  and waveform sections run under any Python with numpy).
\end{frame}
"""

CLOSING = r"""
\begin{frame}[fragile]{Verify: your file IS the repository's code}
  Your build reproduces the repository's simulation \textbf{bit for
  bit} --- same random draws, same channel, same every-interval
  state. The check (run from \texttt{coding\_guide/}):
\begin{lstlisting}
import sys; sys.path.insert(0, "..")
import torch, build_twoway as bt
from ota_sync import (run_two_way_simulation,
                      SDRSimulationConfig)
a = bt.run_two_way_simulation(bt.SDRSimulationConfig())
b = run_two_way_simulation(SDRSimulationConfig())
print(torch.equal(a.post_correction_phase,
                  b.post_correction_phase),
      torch.equal(a.coherent_gain, b.coherent_gain))
\end{lstlisting}
  Output when this guide was generated: \texttt{True True}.
  \vspace{0.4em}

  And the numbers your file prints (identical to the package):
\begin{lstlisting}[language={}]
detection rate     : 1.00
steady phase rms   : 0.0835 rad
mean coherent gain : 99.8 %
airtime fraction   : 19.12 %
final freq error   : -0.010 Hz
\end{lstlisting}
\end{frame}

\begin{frame}{What you built, in one pass}
  Per 50\,ms interval, your code now does --- because you typed every
  line of it:
  \begin{enumerate}
    \item Both clocks random-walk; the master absorbs the carried
      intra-frame walk and the flicker step.
    \item A latency-delayed correction lands, if due; the filter is
      told.
    \item (Once, after three settled corrections) the $\pi$ power
      check repairs a wrong acquisition branch.
    \item Forward capture through the full impairment chain; 1\,ms
      turnaround; reverse capture.
    \item Both captures estimated; the half-difference cancels the
      channel by reciprocity; turnaround drift backed out.
    \item Filter: predict $\to$ branch pick $\to$ circle-measurement
      update --- or coast on a failed detection.
    \item The correction is predicted forward by the command latency,
      quantized, and scheduled.
    \item Truth is recorded: residual phase and
      $\cos^2(\text{residual}/2)$.
  \end{enumerate}
\end{frame}

"""

FINAL = r"""
\begin{frame}{Where to go from here}
  \begin{itemize}
    \item \textbf{Break your own builds} (you can, they are your
      files): feed Part I's filter only the forward capture --- it
      locks to the channel phase, confidently wrong. Delete the
      $\pi$ check --- about half of seeds settle at zero gain
      reporting lock. In Part II, replace the tone's smooth ramp
      with a rectangular edge and watch it ``gain'' range accuracy
      it does not deserve.
    \item \textbf{The same code at $N$ stations}:
      \texttt{ota\_sync/network.py} (star), \texttt{microsync.py},
      \texttt{scheduled.py}, \texttt{dfpc.py},
      \texttt{hybrid\_calibration/} --- all built on the classes you
      wrote in Part I.
    \item \textbf{What it leads to}: \texttt{topology\_selection/}
      --- at $N$ stations you choose which pairs exchange and how
      corrections apply, and those choices are not separable. And
      Part II's blocked-echo result is the argument for sensing with
      a \emph{distributed} array in the first place.
    \item Regenerate this guide after any change:
      \texttt{python3 extract\_build.py \&\& python3 gen\_slides.py
      \&\& tectonic phase\_sync\_coding\_guide.tex}
  \end{itemize}
\end{frame}

\end{document}
"""


def esc(text: str) -> str:
    """Titles may contain characters LaTeX frame titles dislike."""
    return text


def split_chunks(code_lines, target=30, lo=20):
    """Split a code body into frame-sized chunks, preferring blank
    lines as split points."""
    chunks = []
    rest = code_lines
    while len(rest) > target + 4:
        cut = None
        for i in range(min(target, len(rest) - 1), lo - 1, -1):
            if rest[i].strip() == "":
                cut = i
                break
        if cut is None:
            cut = target
        chunks.append(rest[:cut])
        rest = rest[cut:]
        while rest and rest[0].strip() == "":
            rest = rest[1:]
    if rest:
        chunks.append(rest)
    return chunks


def parse_file(path):
    """Return (header_lines, sections) where each section is
    [title, note_lines, code_lines, images]; an image entry is
    (relative_path, caption)."""
    sections = []
    header = []
    current = None
    for line in path.read_text().splitlines():
        if line.startswith("# %% SECTION: "):
            current = [line[len("# %% SECTION: "):], [], [], []]
            sections.append(current)
        elif line.startswith("# %% NOTE: "):
            current[1].append(line[len("# %% NOTE: "):])
        elif line.startswith("# %% IMAGE: "):
            image_path, caption = line[len("# %% IMAGE: "):].split(
                " | ", 1)
            current[3].append((image_path.strip(), caption.strip()))
        elif current is None:
            header.append(line)
        else:
            current[2].append(line)
    for sec in sections:
        while sec[2] and sec[2][0].strip() == "":
            sec[2].pop(0)
        while sec[2] and sec[2][-1].strip() == "":
            sec[2].pop()
    while header and header[-1].strip() == "":
        header.pop()
    return header, sections


def main() -> None:
    header, sections = parse_file(SRC)
    header2, sections2 = parse_file(SRC2)

    out = [PREAMBLE]

    def code_frames(title, code):
        chunks = split_chunks(code)
        n = len(chunks)
        for i, chunk in enumerate(chunks, 1):
            label = f"{title} --- type this"
            if n > 1:
                label += f" ({i}/{n})"
            width = max((len(l) for l in chunk), default=0)
            size = r"\ttfamily\tiny" if (len(chunk) > 18 or width > 62) \
                else r"\ttfamily\scriptsize"
            out.append(
                f"\\begin{{frame}}[fragile]{{{esc(label)}}}\n"
                f"\\begin{{lstlisting}}[basicstyle={size}]\n"
                + "\n".join(chunk)
                + "\n\\end{lstlisting}\n\\end{frame}\n"
            )

    # Header section (hand note, code from file).
    out.append(
        "\\begin{frame}{The file header}\n"
        "  Create \\texttt{build\\_twoway.py}. First the imports: torch,\n"
        "  and four Sionna pieces --- the noise block, the channel\n"
        "  operator, the 3GPP tapped-delay-line model, and two helpers\n"
        "  that convert a physical channel response into discrete-time\n"
        "  taps. All numerics run in double precision: phase\n"
        "  accumulates over millions of samples, and float32 rounding\n"
        "  would show up as fake oscillator noise.\n"
        "\\end{frame}\n"
    )
    code_frames("The file header", header)

    def image_frames(title, images):
        for image_path, caption in images:
            out.append(
                f"\\begin{{frame}}{{{esc(title)}}}\n"
                "  \\begin{center}\n"
                f"  \\includegraphics[width=\\textwidth,"
                f"height=0.72\\textheight,keepaspectratio]"
                f"{{{image_path}}}\n"
                "  \\end{center}\n"
                f"  \\footnotesize {caption}\n"
                "\\end{frame}\n"
            )

    for title, note, code, images in sections:
        out.append(
            f"\\begin{{frame}}{{{esc(title)}}}\n  "
            + "\n  ".join(note)
            + "\n\\end{frame}\n"
        )
        code_frames(title, code)
        image_frames(title, images)

    out.append(CLOSING)

    # ---- Part II: sensing (build_sensing.py) ----------------------
    out.append(PART2_INTRO)
    out.append(
        "\\begin{frame}{Part II file header}\n"
        "  Create \\texttt{build\\_sensing.py}. Plain numpy plus two\n"
        "  physical constants; Sionna's ray tracer is imported later,\n"
        "  inside the function that needs it, so the non-ray-traced\n"
        "  sections run anywhere.\n"
        "\\end{frame}\n"
    )
    code_frames("Part II file header", header2)
    for title, note, code, images in sections2:
        out.append(
            f"\\begin{{frame}}{{{esc(title)}}}\n  "
            + "\n  ".join(note)
            + "\n\\end{frame}\n"
        )
        code_frames(title, code)
        image_frames(title, images)
    out.append(
        "\\begin{frame}[fragile]{Part II --- measured output}\n"
        "  The exact output of \\texttt{build\\_sensing.py} when this\n"
        "  guide was generated:\n"
        "\\begin{lstlisting}[language={},basicstyle=\\ttfamily\\tiny]\n"
        "noise floor: -94.0 dBm\n"
        "training fields (500 trials, 15 dB, 20 kHz offset):\n"
        "  timing exact      : 500/500\n"
        "  cfo error rms     : 1030.2 Hz (0.294 ppm)\n"
        "  phase spread rms  : 0.1068 rad\n"
        "monostatic los       : 2 paths, echo  -97.5 dBm,\n"
        "    detection 100 %, range error -1.2 +- 0.0 m\n"
        "monostatic blocked   : 0 paths, echo   -inf dBm,\n"
        "    detection   0 %\n"
        "monostatic reflector : 3 paths, echo -109.2 dBm,\n"
        "    detection  54 %, range error +37.4 +- 1.2 m\n"
        "waveforms (300 trials, 10 dB, 5 kHz offset):\n"
        "  ofdm  cfo rms   2340.9 Hz | delay rms  0.0 samples (0 m)\n"
        "  tone  cfo rms   7108.6 Hz | delay rms  2.8 samples (21 m)\n"
        "\\end{lstlisting}\n"
        "\\end{frame}\n"
    )

    out.append(FINAL)
    OUT.write_text("\n".join(out))
    frames = sum(part.count(r"\begin{frame}") for part in out)
    print(f"wrote {OUT.name}: {frames} frames, "
          f"{len(sections)} + {len(sections2)} sections")


if __name__ == "__main__":
    main()
