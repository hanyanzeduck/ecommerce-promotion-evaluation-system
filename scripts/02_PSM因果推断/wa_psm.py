import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ==========================================
# wa_psm.py — 倾向得分匹配 (PSM) 因果推断
# ==========================================

# ==========================================
# 1. 数据准备：只保留方案1 和 方案2
# ==========================================
df = pd.read_csv(os.environ.get('WA_DATA_PATH', 'data/raw/WA_Marketing-Campaign.csv'))

# 门店级聚合（4周均值）
store_df = df.groupby(
 ['LocationID', 'MarketID', 'MarketSize', 'AgeOfStore', 'Promotion']
)['SalesInThousands'].mean().reset_index()
store_df.rename(columns={'SalesInThousands': 'AvgSales'}, inplace=True)

# 市场规模转为虚拟变量（以 Large 为基准）
store_df = pd.get_dummies(store_df, columns=['MarketSize'], drop_first=False)
# 生成：MarketSize_Large / MarketSize_Medium / MarketSize_Small

# 只保留方案1 vs 方案2
df_12 = store_df[store_df['Promotion'].isin([1, 2])].copy()
df_12['treatment'] = (df_12['Promotion'] == 1).astype(int)
# treatment=1 → 方案1（实验组）
# treatment=0 → 方案2（对照组）

print(f"方案1门店数: {df_12['treatment'].sum()}")
print(f"方案2门店数: {(df_12['treatment']==0).sum()}")

# ==========================================
# 2. 协变量矩阵（用于估算倾向得分）
# ==========================================
covariates = ['AgeOfStore', 'MarketSize_Medium', 'MarketSize_Small']
# 注意：去掉 MarketSize_Large（避免多重共线性，它是基准组）

X = df_12[covariates].values
T = df_12['treatment'].values
Y = df_12['AvgSales'].values

# ==========================================
# 3. 估算倾向得分（逻辑回归）
# ==========================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lr = LogisticRegression(random_state=42, max_iter=500)
lr.fit(X_scaled, T)
df_12['ps'] = lr.predict_proba(X_scaled)[:, 1] # P(treatment=1 | X)

print("\n=== 倾向得分分布 ===")
print(df_12.groupby('treatment')['ps'].describe().round(4))

# ==========================================
# 4. 共同支撑域（Common Support）检验
# ==========================================
ps_treated = df_12[df_12['treatment'] == 1]['ps']
ps_control = df_12[df_12['treatment'] == 0]['ps']

support_min = max(ps_treated.min(), ps_control.min())
support_max = min(ps_treated.max(), ps_control.max())
print(f"\n共同支撑域: [{support_min:.4f}, {support_max:.4f}]")

# 剔除不在共同支撑域内的门店
df_support = df_12[
 (df_12['ps'] >= support_min) & (df_12['ps'] <= support_max)
].copy()
print(f"剔除后门店数: {len(df_support)}")

# ==========================================
# 5. 最近邻匹配（1:1，有放回）
# ==========================================
treated = df_support[df_support['treatment'] == 1].reset_index(drop=True)
control = df_support[df_support['treatment'] == 0].reset_index(drop=True)

# 用 PS 进行最近邻匹配
nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
nn.fit(control[['ps']].values)
distances, indices = nn.kneighbors(treated[['ps']].values)

matched_control = control.iloc[indices.flatten()].reset_index(drop=True)
matched_treated = treated.copy()

# 合并匹配结果
matched_df = pd.concat([
 matched_treated.assign(group='treated'),
 matched_control.assign(group='control')
], ignore_index=True)

print(f"\n匹配后样本量: {len(matched_treated)} 对")

# ==========================================
# 6. 平衡性检验（匹配前后协变量均值对比）
# ==========================================
print("\n=== 协变量平衡性检验 ===")
print(f"{'变量':<20} {'匹配前_差异':>12} {'匹配后_差异':>12} {'改善':>8}")
print("-" * 56)

for cov in covariates:
 # 匹配前
 before_t = df_12[df_12['treatment']==1][cov].mean()
 before_c = df_12[df_12['treatment']==0][cov].mean()
 before_diff = abs(before_t - before_c)

 # 匹配后
 after_t = matched_treated[cov].mean()
 after_c = matched_control[cov].mean()
 after_diff = abs(after_t - after_c)

 improve = "✓ 改善" if after_diff < before_diff else "✗ 未改善"
 print(f"{cov:<20} {before_diff:>12.4f} {after_diff:>12.4f} {improve:>8}")

# ==========================================
# 7. ATT 估算（匹配后 t 检验）
# ==========================================
sales_treated = matched_treated['AvgSales'].values
sales_control = matched_control['AvgSales'].values

ATT = sales_treated.mean() - sales_control.mean()
t_stat, p_val = stats.ttest_rel(sales_treated, sales_control) # 配对 t 检验

# 95% 置信区间
diff = sales_treated - sales_control
ci_low = ATT - 1.96 * diff.std() / np.sqrt(len(diff))
ci_high = ATT + 1.96 * diff.std() / np.sqrt(len(diff))

print("\n=== PSM 因果效应估计（ATT）===")
print(f"实验组均值（方案1）: {sales_treated.mean():.2f} 千美元/周")
print(f"对照组均值（方案2）: {sales_control.mean():.2f} 千美元/周")
print(f"ATT（平均处理效应）: {ATT:+.2f} 千美元/周")
print(f"95% 置信区间: [{ci_low:.2f}, {ci_high:.2f}]")
print(f"配对 t 检验: t={t_stat:.3f}, p={p_val:.4f}")
print(f"结论: {'方案1效果显著优于方案2 ✓' if p_val < 0.05 else '差异不显著'}")

# ==========================================
# 8. 可视化：三合一图
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 图1：匹配前倾向得分分布
axes[0].hist(ps_treated, bins=15, alpha=0.6, label='Promo 1 (treated)',
 color='#5470c6', density=True)
axes[0].hist(ps_control, bins=15, alpha=0.6, label='Promo 2 (control)',
 color='#91cc75', density=True)
axes[0].set_title('Before Matching — PS Distribution')
axes[0].set_xlabel('Propensity Score')
axes[0].set_ylabel('Density')
axes[0].legend()

# 图2：匹配后倾向得分分布
axes[1].hist(matched_treated['ps'], bins=12, alpha=0.6,
 label='Promo 1 (treated)', color='#5470c6', density=True)
axes[1].hist(matched_control['ps'], bins=12, alpha=0.6,
 label='Promo 2 (control)', color='#91cc75', density=True)
axes[1].set_title('After Matching — PS Distribution')
axes[1].set_xlabel('Propensity Score')
axes[1].set_ylabel('Density')
axes[1].legend()

# 图3：匹配后销售额对比（配对箱线图）
bp_data = [sales_treated, sales_control]
bp = axes[2].boxplot(bp_data, patch_artist=True,
 labels=['Promo 1 (treated)', 'Promo 2 (control)'])
bp['boxes'][0].set_facecolor('#5470c6'); bp['boxes'][0].set_alpha(0.7)
bp['boxes'][1].set_facecolor('#91cc75'); bp['boxes'][1].set_alpha(0.7)
axes[2].set_title(f'Matched Sales Comparison\nATT = {ATT:+.2f}K, p = {p_val:.4f}')
axes[2].set_ylabel('Avg Weekly Sales (K$)')
axes[2].axhline(sales_treated.mean(), color='#5470c6',
 linestyle='--', alpha=0.5, linewidth=1)
axes[2].axhline(sales_control.mean(), color='#91cc75',
 linestyle='--', alpha=0.5, linewidth=1)

plt.tight_layout()
os.makedirs(os.environ.get('WA_OUTPUT_DIR', 'outputs/wa'), exist_ok=True)
output_path = os.path.join(os.environ.get('WA_OUTPUT_DIR', 'outputs/wa'), 'wa_psm_result.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nPSM 图表已保存为 {output_path}")
print("\n=== PSM 分析完成 ===")
