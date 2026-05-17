
## フォルマント推定

### フォルマントとは

```
## kernel

声道の共鳴周波数の推定。「どの周波数帯が声道形状によって増幅されているか」を数値化する操作。

## diag

**軸1：フォルマントとは**

人間が発声するとき、声帯が振動して生成した音が声道（口腔・咽頭）を通る。声道は形状によって特定の周波数を共鳴・増幅する。その共鳴ピーク周波数をフォルマントと呼ぶ。

低い方から順にF1, F2, F3, F4と番号をつける。

```
声帯振動 → 声道フィルタ → 口から放射
（音源）    （共鳴器）      （出力）
```

**軸2：各フォルマントが持つ意味**

| フォルマント | 主な対応 | 概算範囲 |
|------------|---------|---------|
| F1 | 口の開き具合 | 300-900Hz |
| F2 | 舌の前後位置 | 700-2500Hz |
| F3 | 個人差・音色 | 1800-3500Hz |
| F4 | Singer's Formant関連 | 2500-4000Hz |

**軸3：今回の文脈での意義**

「2k-4kHz帯にエネルギーが多い」という観測に対して：

- 帯域エネルギー分析：**結果**のみを見ている
- フォルマント推定：**なぜその帯域にエネルギーがあるか**の構造を見る

F3またはF4が2500-3500Hzに位置していれば「声道共鳴がSinger's Formant帯を作っている」という因果が確認できる。

**軸4：推定方法（LPC）**

スクリプト内のLPCは線形予測符号化。音声波形から「声道フィルタの特性」を逆算する手法。フォルマント周波数＝そのフィルタの極の周波数として取り出す。

## residue

- フォルマントはf0（基音）とは独立した変数
- f0が変わっても声道形状が同じならF1-F4の位置は変わらない
- これが「音域が変わっても2k-4kHz帯が維持される」現象の構造的説明になりうる

```

### 4ファイル分析

```python
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

```

### result
```
(.venv) (base) ume_chan@Lycoris:~/python_app/Sound_Analysis$ python analyze.py

=== Human (1130) ===
f0: median=380.5Hz  IQR=[285.1, 437.1]
帯域エネルギー:
  0-200Hz: 3.33%
  200-500Hz: 29.93%
  500-1000Hz: 38.90%
  1000-2000Hz: 19.18%
  2000-4000Hz: 6.68%
  4000-8000Hz: 1.69%
  8000-+Hz: 0.29%
フォルマント推定:
  F1: 896.5Hz
  F2: 3045.1Hz
  F3: 6222.8Hz
  F4: 11003.3Hz

=== Vocaloid (1223) ===
f0: median=452.5Hz  IQR=[400.9, 563.6]
帯域エネルギー:
  0-200Hz: 1.14%
  200-500Hz: 23.73%
  500-1000Hz: 53.92%
  1000-2000Hz: 16.33%
  2000-4000Hz: 3.77%
  4000-8000Hz: 0.96%
  8000-+Hz: 0.15%
フォルマント推定:
  F1: 830.9Hz
  F2: 3025.6Hz
  F3: 6439.9Hz
  F4: 11029.2Hz

=== Mid (1251) ===
f0: median=349.0Hz  IQR=[286.7, 434.6]
帯域エネルギー:
  0-200Hz: 4.24%
  200-500Hz: 29.69%
  500-1000Hz: 43.25%
  1000-2000Hz: 17.78%
  2000-4000Hz: 3.77%
  4000-8000Hz: 1.09%
  8000-+Hz: 0.17%
フォルマント推定:
  F1: 818.9Hz
  F2: 3131.7Hz
  F3: 6875.3Hz
  F4: 11318.9Hz

=== Lemon (1319) ===
f0: median=238.3Hz  IQR=[197.0, 270.6]
帯域エネルギー:
  0-200Hz: 5.66%
  200-500Hz: 35.95%
  500-1000Hz: 34.91%
  1000-2000Hz: 15.01%
  2000-4000Hz: 7.31%
  4000-8000Hz: 0.94%
  8000-+Hz: 0.22%
フォルマント推定:
  F1: 794.7Hz
  F2: 2927.2Hz
  F3: 6749.0Hz
  F4: 10705.7Hz

→ output_formants.png 出力済み
```