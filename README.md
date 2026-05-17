# Sound_Analysis

Personal acoustic analysis project investigating the cause of microphone feedback (howling) during karaoke.

## Background

When singing at karaoke, a high-pitched metallic sound ("keen") occurs consistently — but only for this vocalist, not others using the same equipment. This repository documents the investigation.

## Equipment

- **Microphone**: Razer gaming headset (VRChat use, unidirectional condenser-type boom mic)
- **Recording format**: m4a (AAC 192kbps, 48kHz stereo) → converted to WAV for analysis
- **Analysis environment**: Lycoris (WSL2, Ubuntu 24.04, RTX 4070), Python 3.13 + librosa 0.11.0

## Samples

| File | Content | f0 median | 2k-4kHz energy |
|------|---------|-----------|----------------|
| sample_202605171130 | Vocal (human-range song, a cappella) | 380.5Hz | 6.68% |
| sample_202605171223 | Vocal (vocaloid original key, a cappella) | 452.5Hz | 3.77% |
| sample_202605171251 | Vocal (mid-range song, a cappella) | 349.0Hz | 3.77% |
| sample_202605171319 | Vocal (Lemon / Kenshi Yonezu, a cappella) | 238.3Hz | 7.31% |
| sample_202605171500 | Spoken vowels: a-i-u-e-o (spoken voice) | — | — |
| sample_202605171600_u | Vowel "u" re-recording | — | — |
| sample_202605171600_o | Vowel "o" re-recording | — | — |

All samples are a cappella (no accompaniment). Confirmed that observed spectral characteristics reflect vocal production, not instrumental content.

## Key Findings

### 1. Howling band is independent of f0

The 2k-4kHz band energy (Singer's Formant region) remains consistently elevated regardless of pitch:

- f0 range across samples: 238Hz–452Hz
- 2k-4kHz energy: 6.68–7.44% (human-range songs), vs 2.66–3.77% (vocaloid)
- Lowering pitch does not reduce Singer's Formant band energy

### 2. F2 shifts into Singer's Formant band during singing

LPC-based formant estimation (order=50):

| Condition | F2 range | Singer's Formant band (2.5–3.5kHz) |
|-----------|----------|--------------------------------------|
| Spoken voice (a, i, e) | 1096–2033Hz | Outside |
| Singing (all 4 samples) | 2927–3132Hz | Inside |

F2 position is stable across singing samples despite f0 variation — indicating a consistent vocal tract shape during singing.

### 3. Mechanism

```
Singing mode activated
→ Vocal tract shape shifts (larynx lowered, soft palate raised)
→ F2 locks into 2900–3130Hz (Singer's Formant band)
→ Microphone captures elevated 2k-4kHz energy
→ Amplifier → Speaker → Microphone loop
→ Loop gain ≥ 1 at Singer's Formant frequencies
→ Howling ("keen")
```

Vocaloid synthesis lacks the vocal tract resonance structure that produces Singer's Formant, explaining why vocaloid-key singing produces less feedback.

### 4. Background context

Vocalist trained in Italian opera during university. Singer's Formant production appears retained as procedural/motor memory, activated automatically in singing mode. Karaoke systems are calibrated for untrained voices with lower Singer's Formant energy — this creates a systematic mismatch.

## Practical Countermeasure

Reduce microphone gain by 2–3 steps at the start of each karaoke session. Target EQ notch: **2.9–3.1kHz**.

## Analysis Scripts

| Script | Purpose |
|--------|---------|
| `analyze.py` | Band energy ratio + LPC formant estimation across 4 singing samples |
| `vowel_analysis.py` | Segmentation and per-vowel formant analysis (spoken voice) |
| `vowel_uo.py` | Single-file analysis for "u" and "o" re-recordings |
| `vowel_plot.py` | Vowel chart (F1-F2 scatter) vs Japanese typical values |

## Limitations

- Recording equipment: gaming headset with voice-optimized EQ (high-frequency emphasis). May introduce bias in absolute spectral values.
- "u" and "o" vowel LPC estimation failed at order=50; fell back to order=16. F1/F2 values for these vowels are unreliable.
- Singing F1 not measured (only F2 available from LPC on singing samples).
- Sample size: single vocalist, single session.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install librosa scipy matplotlib numpy
# m4a to wav conversion
ffmpeg -i input.m4a output.wav
```
