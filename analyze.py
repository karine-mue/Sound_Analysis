# analyze.py
# ~/python_app/Sound_Analysis/ に置いて実行

import numpy as np
import librosa
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("./data")
SAMPLES = [
    ("1130", "Human"),
    ("1223", "Vocaloid"),
    ("1251", "Mid"),
    ("1319", "Lemon"),
]

BANDS = [
    (0, 200),
    (200, 500),
    (500, 1000),
    (1000, 2000),
    (2000, 4000),
    (4000, 8000),
    (8000, None),
]

def band_energy_ratio(S, freqs, fmin, fmax):
    mask = (freqs >= fmin) & (freqs < (fmax if fmax else freqs[-1] + 1))
    total = np.sum(S**2)
    return np.sum(S[mask]**2) / total * 100 if total > 0 else 0

def lpc_formants(y, sr, order=12, n_frames=200):
    """LPCによるフォルマント推定（F1-F4）"""
    hop = len(y) // n_frames
    formants = {f"F{i+1}": [] for i in range(4)}
    
    for i in range(n_frames):
        frame = y[i*hop:(i+1)*hop]
        if len(frame) < order + 1:
            continue
        # LPC係数
        a = librosa.lpc(frame, order=order)
        # 根を求めてフォルマント周波数に変換
        roots = np.roots(a)
        roots = roots[np.imag(roots) >= 0]
        angles = np.angle(roots)
        freqs_f = angles * (sr / (2 * np.pi))
        freqs_f = np.sort(freqs_f[freqs_f > 80])
        for j in range(min(4, len(freqs_f))):
            formants[f"F{j+1}"].append(freqs_f[j])
    
    return {k: np.median(v) for k, v in formants.items() if v}

results = {}

for tag, label in SAMPLES:
    path = DATA_DIR / f"sample_20260517{tag}.wav"
    y, sr = librosa.load(path, sr=None, mono=True)
    
    # スペクトル
    S = np.abs(librosa.stft(y, n_fft=4096))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    S_mean = S.mean(axis=1)
    
    # 帯域エネルギー
    band_ratios = {}
    for fmin, fmax in BANDS:
        key = f"{fmin}-{fmax if fmax else '+'}"
        band_ratios[key] = band_energy_ratio(S_mean, freqs, fmin, fmax or 24000)
    
    # f0推定
    f0, _, _ = librosa.pyin(y, fmin=80, fmax=900, sr=sr)
    f0_valid = f0[~np.isnan(f0)]
    
    # LPCフォルマント
    formants = lpc_formants(y, sr)
    
    results[tag] = {
        "label": label,
        "band_ratios": band_ratios,
        "f0_median": np.median(f0_valid),
        "f0_p25": np.percentile(f0_valid, 25),
        "f0_p75": np.percentile(f0_valid, 75),
        "formants": formants,
    }
    print(f"\n=== {label} ({tag}) ===")
    print(f"f0: median={results[tag]['f0_median']:.1f}Hz  IQR=[{results[tag]['f0_p25']:.1f}, {results[tag]['f0_p75']:.1f}]")
    print("帯域エネルギー:")
    for k, v in band_ratios.items():
        print(f"  {k}Hz: {v:.2f}%")
    print("フォルマント推定:")
    for k, v in formants.items():
        print(f"  {k}: {v:.1f}Hz")

# 可視化：2k-4kHz帯 + フォルマントの横断比較
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

labels = [results[t]["label"] for t, _ in SAMPLES]
band_2k4k = [results[t]["band_ratios"]["2000-4000"] for t, _ in SAMPLES]
f0_meds = [results[t]["f0_median"] for t, _ in SAMPLES]

ax = axes[0]
bars = ax.bar(labels, band_2k4k, color=["#4C72B0","#DD8452","#55A868","#C44E52"])
#ax.set_title("2k-4kHz エネルギー比率（Singer's Formant帯）")
ax.set_title("2k–4kHz Energy Ratio (Singer's Formant Band)")
ax.set_ylabel("%")
for bar, val in zip(bars, band_2k4k):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.1, f"{val:.2f}%", ha="center")

ax = axes[1]
for i, (tag, _) in enumerate(SAMPLES):
    f = results[tag]["formants"]
    ks = sorted(f.keys())
    vals = [f[k] for k in ks]
    ax.plot(ks, vals, marker="o", label=results[tag]["label"])
ax.axhspan(2500, 3500, alpha=0.1, color="red", label="Singer's Formant Band")
#ax.set_title("フォルマント推定（F1-F4）")
ax.set_title("Formant Estimation (F1–F4)")
ax.set_ylabel("Hz")
ax.legend()

plt.tight_layout()
plt.savefig("./output_formants.png", dpi=150)
print("\n→ output_formants.png 出力済み")