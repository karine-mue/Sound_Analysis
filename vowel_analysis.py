# vowel_analysis.py
# ~/python_app/Sound_Analysis/ に置いて実行

import numpy as np
import librosa
import matplotlib.pyplot as plt
from pathlib import Path

WAV_PATH = Path("./data/sample_202605171500.wav")
OUTPUT_DIR = Path("./data/pre-process")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- 無音区間で分割 ---
y, sr = librosa.load(WAV_PATH, sr=None, mono=True)

intervals = librosa.effects.split(y, top_db=30, frame_length=2048, hop_length=512)
print(f"検出セグメント数: {len(intervals)}")
for i, (start, end) in enumerate(intervals):
    duration = (end - start) / sr
    print(f"  seg{i:02d}: {start/sr:.2f}s - {end/sr:.2f}s  ({duration:.2f}s)")


# --- LPCフォルマント推定（order=50） ---
# def lpc_formants(segment, sr, order=50):
#     a = librosa.lpc(segment, order=order)
#     roots = np.roots(a)
#     roots = roots[np.imag(roots) >= 0]
#     angles = np.angle(roots)
#     freqs = np.sort(angles * (sr / (2 * np.pi)))
#     freqs = freqs[freqs > 80]
#     result = {}
#     for j in range(min(4, len(freqs))):
#         result[f"F{j+1}"] = freqs[j]
#     return result

# --- LPCフォルマント推定（order=50）改　母音のうち「う」と「お」が落ちる問題。 ---
# 原因は不明だが、segmentのRMSが低いフレームが多いとlpcの係数にNaNやInfが入ることがある。
# そこで、segmentの中央50%を取った後、さらにRMSが低いフレームを除去してからlpc_formantsを呼び出すように変更してみる。
# lpc_formantsの中でinfが出ているはずなので、関数内部にtry/exceptを追加してどのステップで死んでいるか確認してみる。
# def lpc_formants(segment, sr, order=50):
#     a = librosa.lpc(segment, order=order)
#     print(f"    lpc done, a has inf: {np.any(np.isinf(a))}, nan: {np.any(np.isnan(a))}")
#     roots = np.roots(a)
#     print(f"    roots done, has inf: {np.any(np.isinf(roots))}, nan: {np.any(np.isnan(roots))}")
#     roots = roots[np.imag(roots) >= 0]
#     angles = np.angle(roots)
#     freqs = np.sort(angles * (sr / (2 * np.pi)))
#     freqs = freqs[freqs > 80]
#     result = {}
#     for j in range(min(4, len(freqs))):
#         result[f"F{j+1}"] = freqs[j]
#     return result

# --- LPCフォルマント推定（order=50）改2　母音のうち「う」と「お」が落ちる問題。 ---
# librosa.lpcがNaNを返している。
# seg02（う）とseg04（お）でのみa has nan: True。lpc自体が死んでいる。
# librosa.lpcはorder=50で長いセグメントに対してlevinson-durbin再帰を使うが、特定の音声特性（うとおは唇を丸めた円唇母音）で自己相関行列が数値的に特異になるケースがある。
# 対処：orderを下げてフォールバックする
def lpc_formants(segment, sr, order=50):
    # orderを下げながらリトライ
    for o in [order, 30, 20, 16]:
        a = librosa.lpc(segment, order=o)
        if not (np.any(np.isnan(a)) or np.any(np.isinf(a))):
            break
    else:
        raise ValueError("LPC failed at all orders")
    
    roots = np.roots(a)
    roots = roots[np.imag(roots) >= 0]
    angles = np.angle(roots)
    freqs = np.sort(angles * (sr / (2 * np.pi)))
    freqs = freqs[freqs > 80]
    result = {}
    for j in range(min(4, len(freqs))):
        result[f"F{j+1}"] = freqs[j]
    return result

# --- 各セグメントを解析 ---
VOWEL_LABELS = ["a", "i", "u", "e", "o"]  # 録音順に合わせて変更

seg_results = []
for i, (start, end) in enumerate(intervals):
    segment = y[start:end]
    # 中央50%を使用（立ち上がり・立ち下がりを除外）
    trim_start = len(segment) // 4
    trim_end = 3 * len(segment) // 4
    core = segment[trim_start:trim_end]

    order = 50

    # 中央50%を取った後、さらにRMSが低いフレームを除去
    frame_length = 2048
    hop = 512
    rms = librosa.feature.rms(y=core, frame_length=frame_length, hop_length=hop)[0]
    threshold = np.percentile(rms, 30)  # 下位30%を除去
    active_frames = np.where(rms >= threshold)[0]

    if len(active_frames) == 0:
        print(f"seg{i:02d}: no active frames, skip")
        continue

    # active frameに対応するサンプルだけ使う
    start_sample = active_frames[0] * hop
    end_sample = min(active_frames[-1] * hop + frame_length, len(core))
    core = core[start_sample:end_sample]
    
    if len(core) < order + 10:
        print(f"seg{i:02d}: too short, skip")
        continue
    
    try:
        #order = 50

        # lpc_formants呼び出し直前に追加(RMSが低いフレームを除去した後のcoreに対して)
        core = core / (np.max(np.abs(core)) + 1e-8)  # 正規化

        # coreのRMS統計を出力（lpc_formants呼び出し前に追加）
        print(f"  core length: {len(core)/sr:.2f}s  RMS max: {np.max(np.abs(core)):.4f}  NaN: {np.any(np.isnan(core))}  Inf: {np.any(np.isinf(core))}")

        formants = lpc_formants(core, sr, order=order)
        f0, _, _ = librosa.pyin(core, fmin=80, fmax=900, sr=sr)
        f0_valid = f0[~np.isnan(f0)]
        f0_med = np.median(f0_valid) if len(f0_valid) > 0 else float("nan")
        
        label = VOWEL_LABELS[i] if i < len(VOWEL_LABELS) else f"seg{i:02d}"
        seg_results.append({"label": label, "f0": f0_med, "formants": formants})
        
        print(f"\n[{label}]  f0={f0_med:.1f}Hz")
        for k, v in formants.items():
            marker = " ← Singer's Formant帯" if 2500 <= v <= 3500 else ""
            print(f"  {k}: {v:.1f}Hz{marker}")
    except Exception as e:
        print(f"seg{i:02d}: error - {e}")

# --- 可視化 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# F1-F4 横断
ax = axes[0]
for r in seg_results:
    f = r["formants"]
    ks = sorted(f.keys())
    vals = [f[k] for k in ks]
    ax.plot(ks, vals, marker="o", label=r["label"])
ax.axhspan(2500, 3500, alpha=0.15, color="red", label="Singer's Formant")
ax.set_title("Formant Estimation by Vowel (order=50)")
ax.set_ylabel("Hz")
ax.legend()

# F2のみ比較
ax = axes[1]
labels = [r["label"] for r in seg_results]
f2_vals = [r["formants"].get("F2", float("nan")) for r in seg_results]
bars = ax.bar(labels, f2_vals, color=["#4C72B0","#DD8452","#55A868","#C44E52","#9467bd"])
ax.axhspan(2500, 3500, alpha=0.15, color="red", label="Singer's Formant")
ax.set_title("F2 by Vowel")
ax.set_ylabel("Hz")
ax.legend()
for bar, val in zip(bars, f2_vals):
    if not np.isnan(val):
        ax.text(bar.get_x() + bar.get_width()/2, val + 50, f"{val:.0f}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("./output_vowels.png", dpi=150)
print("\n→ output_vowels.png 出力済み")