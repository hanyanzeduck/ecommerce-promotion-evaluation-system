import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ==========================================
# Step 1：加载数据
# ==========================================
df = pd.read_csv(os.environ.get('WA_DATA_PATH', 'data/raw/WA_Marketing-Campaign.csv'))
print(f"数据形状: {df.shape}")
print(df.head())

# ==========================================
# Step 2：数据清洗与预处理
# ==========================================

# 无缺失值，直接做编码
df['MarketSize_code'] = LabelEncoder().fit_transform(df['MarketSize'])

# 构造 treatment 变量（以促销方案1为基准）
# 论文中把3组分别作为 treated/control 对比
df['treat_1vs2'] = (df['Promotion'] == 1).astype(int) # 方案1 vs 方案2
df['treat_1vs3'] = (df['Promotion'] == 1).astype(int) # 方案1 vs 方案3

# 门店级别聚合（4周平均销售额）
store_df = df.groupby(['LocationID', 'MarketID', 'MarketSize',
 'MarketSize_code', 'AgeOfStore', 'Promotion']
 )['SalesInThousands'].mean().reset_index()
store_df.rename(columns={'SalesInThousands': 'AvgSales'}, inplace=True)

print(f"\n门店级聚合后: {store_df.shape} 行")
print(store_df.head())

# ==========================================
# Step 3：描述性统计 - 各促销方案对比
# ==========================================
print("\n=== 各促销方案平均销售额 ===")
summary = store_df.groupby('Promotion')['AvgSales'].agg(
 ['mean', 'std', 'count', 'median']
).round(2)
print(summary)

# ==========================================
# Step 4：方差分析 (ANOVA) - 判断3组是否有差异
# ==========================================
g1 = store_df[store_df['Promotion'] == 1]['AvgSales']
g2 = store_df[store_df['Promotion'] == 2]['AvgSales']
g3 = store_df[store_df['Promotion'] == 3]['AvgSales']

f_stat, p_value = stats.f_oneway(g1, g2, g3)
print(f"\n=== 单因素方差分析 (ANOVA) ===")
print(f"F统计量: {f_stat:.4f}")
print(f"p值: {p_value:.4f}")
print(f"结论: {'三组促销方案存在显著差异 (p<0.05)' if p_value < 0.05 else '三组促销方案无显著差异'}")

# ==========================================
# Step 5：两两 t 检验（事后检验）
# ==========================================
print("\n=== 两两 t 检验 ===")
pairs = [(1,2), (1,3), (2,3)]
for a, b in pairs:
 ga = store_df[store_df['Promotion'] == a]['AvgSales']
 gb = store_df[store_df['Promotion'] == b]['AvgSales']
 t, p = stats.ttest_ind(ga, gb)
 diff = ga.mean() - gb.mean()
 sig = "**显著**" if p < 0.05 else "不显著"
 print(f"方案{a} vs 方案{b}: 均值差={diff:.2f}千美元, t={t:.3f}, p={p:.4f} → {sig}")

# ==========================================
# Step 6：简单 OLS 回归（控制混淆变量）
# ==========================================
from statsmodels.formula.api import ols
import statsmodels.api as sm

# 控制 MarketSize 和 AgeOfStore 后，Promotion的纯效果
# 以方案3为基准（baseline）
store_df['Promo_1'] = (store_df['Promotion'] == 1).astype(int)
store_df['Promo_2'] = (store_df['Promotion'] == 2).astype(int)

model = ols(
 'AvgSales ~ Promo_1 + Promo_2 + AgeOfStore + MarketSize_code',
 data=store_df
).fit()

print("\n=== OLS 回归结果（控制混淆变量后的促销效果）===")
print(model.summary().tables[1]) # 只打印系数表

# ==========================================
# Step 7：计算 ROI 相关指标
# ==========================================
# 以方案2作为基准（最差），计算方案1和3的增量销售额
baseline_sales = g2.mean()
print(f"\n=== ROI 分析（以方案2为基准）===")
for promo, g in [(1, g1), (3, g3)]:
 uplift = g.mean() - baseline_sales
 uplift_pct = uplift / baseline_sales * 100
 print(f"方案{promo}: 相对方案2 增量={uplift:.2f}千美元/店/周, 增幅={uplift_pct:.1f}%")

# ==========================================
# Step 8：可视化
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 图1：各方案销售额箱线图
colors = ['#5470c6', '#91cc75', '#fac858']
data_by_promo = [
 store_df[store_df['Promotion'] == i]['AvgSales'].values
 for i in [1, 2, 3]
]
bp = axes[0].boxplot(data_by_promo, patch_artist=True,
 labels=['Promo 1', 'Promo 2', 'Promo 3'])
for patch, color in zip(bp['boxes'], colors):
 patch.set_facecolor(color)
 patch.set_alpha(0.7)
axes[0].set_title('Sales Distribution by Promotion')
axes[0].set_ylabel('Avg Weekly Sales (K$)')

# 图2：按市场规模分组的促销效果
pivot = store_df.groupby(['MarketSize', 'Promotion'])['AvgSales'].mean().unstack()
pivot.plot(kind='bar', ax=axes[1], color=colors, alpha=0.8)
axes[1].set_title('Sales by Market Size & Promotion')
axes[1].set_ylabel('Avg Weekly Sales (K$)')
axes[1].set_xlabel('Market Size')
axes[1].legend(['Promo 1', 'Promo 2', 'Promo 3'])
axes[1].tick_params(axis='x', rotation=0)

# 图3：周趋势折线图
weekly = df.groupby(['week', 'Promotion'])['SalesInThousands'].mean().unstack()
for i, color in zip([1, 2, 3], colors):
 axes[2].plot(weekly.index, weekly[i], marker='o',
 label=f'Promo {i}', color=color, linewidth=2)
axes[2].set_title('Weekly Sales Trend')
axes[2].set_xlabel('Week')
axes[2].set_ylabel('Avg Sales (K$)')
axes[2].legend()
axes[2].set_xticks([1, 2, 3, 4])

plt.tight_layout()
os.makedirs(os.environ.get('WA_OUTPUT_DIR', 'outputs/wa'), exist_ok=True)
output_path = os.path.join(os.environ.get('WA_OUTPUT_DIR', 'outputs/wa'), 'wa_analysis_result.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n图表已保存为 {output_path}")
print("\n=== 处理完成 ===")
