# vowel_uo.py
# ~/python_app/Sound_Analysis/ に置いて実行

import numpy as np
import librosa

def lpc_formants(segment, sr, order=50):
    for o in [order, 30, 20, 16]:
        a = librosa.lpc(segment, order=o)
        if not (np.any(np.isnan(a)) or np.any(np.isinf(a))):
            print(f"    order={o} で成功")
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

for label, path in [("u", "./data/sample_202605171600_u.wav"),
                    ("o", "./data/sample_202605171600_o.wav")]:
    y, sr = librosa.load(path, sr=None, mono=True)
    
    # 無音除去
    intervals = librosa.effects.split(y, top_db=30)
    if len(intervals) == 0:
        print(f"[{label}] no active segment")
        continue
    # 最長セグメントを使用
    longest = max(intervals, key=lambda x: x[1] - x[0])
    segment = y[longest[0]:longest[1]]
    
    # 中央50%
    t0 = len(segment) // 4
    t1 = 3 * len(segment) // 4
    core = segment[t0:t1]
    
    # RMSフィルタ
    rms = librosa.feature.rms(y=core, frame_length=2048, hop_length=512)[0]
    threshold = np.percentile(rms, 30)
    active = np.where(rms >= threshold)[0]
    if len(active) > 0:
        core = core[active[0]*512 : min(active[-1]*512+2048, len(core))]
    
    # 正規化
    core = core / (np.max(np.abs(core)) + 1e-8)
    
    print(f"\n[{label}]  core={len(core)/sr:.2f}s  RMS max={np.max(np.abs(core)):.4f}")
    
    try:
        formants = lpc_formants(core, sr)
        f0, _, _ = librosa.pyin(core, fmin=80, fmax=900, sr=sr)
        f0_valid = f0[~np.isnan(f0)]
        f0_med = np.median(f0_valid) if len(f0_valid) > 0 else float("nan")
        print(f"f0={f0_med:.1f}Hz")
        for k, v in formants.items():
            marker = " ← SF帯" if 2500 <= v <= 3500 else ""
            print(f"  {k}: {v:.1f}Hz{marker}")
    except Exception as e:
        print(f"error: {e}")