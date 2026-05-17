# vowel_plot.py

import numpy as np
import matplotlib.pyplot as plt

# Karineの実測値（地声）
karine_measured = {
    "a": (698.3, 1095.7),
    "i": (355.9, 1991.2),
    "e": (528.7, 2033.3),
}

# 日本語母音の典型値
jp_typical = {
    "a": (800, 1200),
    "i": (300, 2300),
    "u": (400, 1100),
    "e": (600, 2000),
    "o": (500, 800),
}

# 歌唱サンプルのF2（F1は今回未取得なので推定不能→F2のみ別途表示）
karine_singing_f2 = {
    "Human":    (None, 3045),
    "Vocaloid": (None, 3026),
    "Mid":      (None, 3132),
    "Lemon":    (None, 2927),
}

fig, ax = plt.subplots(figsize=(10, 7))

# 典型値
for vowel, (f1, f2) in jp_typical.items():
    ax.scatter(f2, f1, s=200, c="lightgray", edgecolors="gray", zorder=2)
    ax.annotate(f"{vowel} (ref)", (f2, f1),
                textcoords="offset points", xytext=(8, 4),
                fontsize=9, color="gray")

# Karine地声
for vowel, (f1, f2) in karine_measured.items():
    ax.scatter(f2, f1, s=200, c="#4C72B0", edgecolors="navy", zorder=3)
    ax.annotate(f"{vowel} (Karine)", (f2, f1),
                textcoords="offset points", xytext=(8, 4),
                fontsize=9, color="#4C72B0")

# Singer's Formant帯（F2軸上の縦帯）
ax.axvspan(2500, 3500, alpha=0.12, color="red", label="Singer's Formant (2.5-3.5kHz)")

# 歌唱F2をF1=500固定（暫定）で表示
for label, (_, f2) in karine_singing_f2.items():
    ax.scatter(f2, 500, s=150, c="#DD8452", marker="^",
               edgecolors="darkorange", zorder=3)
    ax.annotate(f"{label}\n(singing F2)", (f2, 500),
                textcoords="offset points", xytext=(6, -20),
                fontsize=8, color="#DD8452")

# 軸設定（母音図の慣例：F2は右から左、F1は下から上）
ax.invert_xaxis()
ax.invert_yaxis()
ax.set_xlabel("F2 (Hz)", fontsize=12)
ax.set_ylabel("F1 (Hz)", fontsize=12)
ax.set_title("Vowel Chart: Karine (spoken) vs Japanese typical vs Singing F2", fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("./output_vowel_chart.png", dpi=150)
print("→ output_vowel_chart.png 出力済み")