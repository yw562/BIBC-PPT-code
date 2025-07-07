import matplotlib.pyplot as plt

# -------- timeline data --------
dates  = ["2024-Q4", "2025-Q2", "2025-Q3", "2025-Q4", "2026-Q1"]
events = [
    "ISO 10218 gap analysis",
    "CE Machinery tech-file submitted",
    "EMC + RED lab tests (TÜV)",
    "ISO 13482 functional-safety FMEA passed",
    "UKCA / CE certificates granted"
]

# -------- plotting --------
plt.figure(figsize=(10, 2.2))          # 稍加宽

plt.hlines(0, 0, len(dates)-1, color="#1565c0", linewidth=2)   # 主线

for idx, (d, e) in enumerate(zip(dates, events)):
    plt.plot(idx, 0, marker='o', markersize=12, color="#1565c0")
    plt.text(idx, 0.25, d,  ha='center', va='bottom', fontsize=10)   # 日期 ↑
    plt.text(idx,-0.25, e, ha='center', va='top',    fontsize=9)    # 事件 ↓

# -------- cosmetic --------
plt.ylim(-0.6, 0.6)      # 给上下留白
plt.axis("off")          # 去掉坐标
plt.tight_layout()

plt.savefig("regulatory_safety_roadmap.png", dpi=300)
print("Saved as regulatory_safety_roadmap.png")
