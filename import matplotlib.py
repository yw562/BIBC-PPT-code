import matplotlib.pyplot as plt

# ---------- 1. 数据 ----------
platforms  = ['Diligent Moxi', 'Keenon Relay', 'Optimus / Figure 01', 'Robolution']
cost_kUSD = [35, 15, 30, 7]          # X 轴
autonomy   = [20, 40, 90, 70]        # Y 轴

# ---------- 2. 绘图 ----------
plt.figure(figsize=(6, 4))

# 先画 3 个竞品：浅灰圆点
for x, y, name in zip(cost_kUSD[:-1], autonomy[:-1], platforms[:-1]):
    plt.scatter(x, y, color='#B0B0B0', s=80)

# 再画 Robolution：蓝色星形
plt.scatter(cost_kUSD[-1], autonomy[-1],
            color='#0070C0', marker='*', s=160, label='Robolution')

# 坐标轴
plt.xlim(0, 40)
plt.ylim(0, 100)
plt.xlabel("Cost (k USD)")
plt.ylabel("Autonomy Score (0–100)")

# 网格与标题
plt.grid(alpha=0.3)
# plt.title("Competitive Landscape")

# # 脚注
# plt.text(0, -8,
#          "*Cost: vendor/media sources; Autonomy score = qualitative scale.",
#          fontsize=8, color='gray')

# ---------- 3. 保存 ----------
plt.tight_layout()
plt.savefig("competitive_scatter.png", dpi=300)
print("Saved → competitive_scatter.png")


# import matplotlib.pyplot as plt
# import pandas as pd

# # -------- 1. 数据 --------
# data = {
#     'Platform': ['Diligent Moxi', 'Keenon Relay', 'Optimus / Figure 01', 'Robolution'],
#     'Cost_kUSD': [35, 15, 30, 7],
#     'Autonomy': [20, 40, 90, 70],
#     'Series': ['Competitor', 'Competitor', 'Competitor', 'Robolution']
# }
# df = pd.DataFrame(data)

# # -------- 2. 绘图 --------
# plt.figure(figsize=(6,4))

# # 先画 3 个竞品（用默认颜色）
# df[df.Series == 'Competitor'].plot.scatter(
#     x='Cost_kUSD', y='Autonomy', s=60,
#     label='Competitors', ax=plt.gca()
# )

# # 再画 Robolution，默认会用第二种颜色并自动加图例
# df[df.Series == 'Robolution'].plot.scatter(
#     x='Cost_kUSD', y='Autonomy', s=140, marker='*',
#     label='Robolution', ax=plt.gca()
# )

# # 坐标轴 & 栅格
# plt.xlim(0, 40)
# plt.ylim(0, 100)
# plt.xlabel("Cost (k USD)")
# plt.ylabel("Autonomy Score (0–100)")
# plt.title("Competitive Landscape")
# plt.grid(alpha=0.3)
# plt.legend(frameon=False)

# # -------- 3. 保存 --------
# plt.tight_layout()
# plt.savefig("competitive_scatter.png", dpi=300)
# print("图已保存为 competitive_scatter.png")
