# ==========================================
# wa_did.py — 双重差分法 (DiD) 因果推断
# 数据: WA_Marketing-Campaign.csv
# Input and output locations are configured through environment variables.
# ==========================================

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── 字体设置（Windows本地运行改为 Microsoft YaHei）──────
# matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
# matplotlib.rcParams['axes.unicode_minus'] = False

DATA_PATH = os.environ.get('WA_DATA_PATH', 'data/raw/WA_Marketing-Campaign.csv')
OUTPUT_DIR = os.environ.get('WA_OUTPUT_DIR', 'outputs/wa')

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# Step 1：数据准备
# ==========================================
print('Step 1: 加载数据...')
df = pd.read_csv(DATA_PATH)

# 只保留方案1（处理组）和方案2（对照组）
df12 = df[df['Promotion'].isin([1, 2])].copy()

# 构造DiD核心变量
# post: week1-2 = 活动前（0），week3-4 = 活动后（1）
df12['treated'] = (df12['Promotion'] == 1).astype(int)
df12['post']    = (df12['week'] >= 3).astype(int)
df12['did']     = df12['treated'] * df12['post']   # 交互项 = DiD估计量

# 市场规模虚拟变量
df12 = pd.get_dummies(df12, columns=['MarketSize'], drop_first=False)
market_dummies = [c for c in df12.columns if 'MarketSize_' in c]
# 去掉 Large 作为基准
if 'MarketSize_Large' in market_dummies:
    market_dummies.remove('MarketSize_Large')

print(f'  样本量: {len(df12)} 条记录')
print(f'  处理组门店数: {df12[df12["treated"]==1]["LocationID"].nunique()}')
print(f'  对照组门店数: {df12[df12["treated"]==0]["LocationID"].nunique()}')

# ==========================================
# Step 2：平行趋势检验（Pre-trend Check）
# ==========================================
print('\nStep 2: 平行趋势检验...')

# 仅看前期（week1-2），对照组和处理组趋势是否相似
pre_data = df12[df12['post'] == 0].copy()
weekly_pre = pre_data.groupby(['week','treated'])['SalesInThousands'].mean().unstack()

slope_t = weekly_pre[1].iloc[1] - weekly_pre[1].iloc[0]
slope_c = weekly_pre[0].iloc[1] - weekly_pre[0].iloc[0]
print(f'  前期处理组周均变化: {slope_t:+.2f}K')
print(f'  前期对照组周均变化: {slope_c:+.2f}K')
print(f'  趋势差异: {abs(slope_t - slope_c):.2f}K  '
      f'{"（趋势相近，平行假设基本成立）" if abs(slope_t - slope_c) < 3 else "（需关注）"}')

# ==========================================
# Step 3：描述性统计（2×2 均值表）
# ==========================================
print('\nStep 3: 2×2 均值表...')

pre_t  = df12[(df12['treated']==1)&(df12['post']==0)]['SalesInThousands'].mean()
post_t = df12[(df12['treated']==1)&(df12['post']==1)]['SalesInThousands'].mean()
pre_c  = df12[(df12['treated']==0)&(df12['post']==0)]['SalesInThousands'].mean()
post_c = df12[(df12['treated']==0)&(df12['post']==1)]['SalesInThousands'].mean()

did_manual = (post_t - pre_t) - (post_c - pre_c)

print(f'\n  {"":12} {"活动前（Week1-2）":>16} {"活动后（Week3-4）":>16} {"变化量":>10}')
print(f'  {"处理组（方案1）":<14} {pre_t:>16.2f}K {post_t:>16.2f}K {post_t-pre_t:>+10.2f}K')
print(f'  {"对照组（方案2）":<14} {pre_c:>16.2f}K {post_c:>16.2f}K {post_c-pre_c:>+10.2f}K')
print(f'  {"差中差（DiD）":<14} {"":>16} {"":>16} {did_manual:>+10.2f}K')

# ==========================================
# Step 4：OLS回归估计DiD
# ==========================================
print('\nStep 4: OLS回归估计DiD...')

# 基础模型
formula_base = 'SalesInThousands ~ treated + post + did'
model_base = smf.ols(formula_base, data=df12).fit()

# 加控制变量（AgeOfStore + 市场规模虚拟变量）
ctrl_vars = ' + '.join(['AgeOfStore'] + market_dummies)
formula_full = f'SalesInThousands ~ treated + post + did + {ctrl_vars}'
model_full = smf.ols(formula_full, data=df12).fit()

print('\n  === 基础DiD模型 ===')
print(f'  DiD系数（did）: {model_base.params["did"]:.4f}K  '
      f'p = {model_base.pvalues["did"]:.4f}  '
      f'{"** 显著" if model_base.pvalues["did"] < 0.05 else "不显著"}')
print(f'  R²: {model_base.rsquared:.4f}')

print('\n  === 加入控制变量的DiD模型 ===')
print(f'  DiD系数（did）: {model_full.params["did"]:.4f}K  '
      f'p = {model_full.pvalues["did"]:.4f}  '
      f'{"** 显著" if model_full.pvalues["did"] < 0.05 else "不显著"}')
print(f'  R²: {model_full.rsquared:.4f}')

print('\n  完整回归系数表（加控制变量版）:')
results_table = pd.DataFrame({
    '系数': model_full.params.round(4),
    '标准误': model_full.bse.round(4),
    't值': model_full.tvalues.round(3),
    'p值': model_full.pvalues.round(4)
})
print(results_table.to_string())

# ==========================================
# Step 5：门店固定效应模型（更严谨）
# ==========================================
print('\nStep 5: 门店固定效应DiD...')

# 添加门店固定效应（吸收门店层面不随时间变化的异质性）
df12['LocationID'] = df12['LocationID'].astype(str)
formula_fe = 'SalesInThousands ~ post + did + C(LocationID)'
model_fe = smf.ols(formula_fe, data=df12).fit()

print(f'  门店固定效应DiD系数: {model_fe.params["did"]:.4f}K  '
      f'p = {model_fe.pvalues["did"]:.4f}  '
      f'{"** 显著" if model_fe.pvalues["did"] < 0.05 else "不显著"}')
print(f'  R²: {model_fe.rsquared:.4f}')

# ==========================================
# Step 6：可视化（4张图）
# ==========================================
print('\nStep 6: 生成可视化...')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

colors = {'treated': '#5470c6', 'control': '#91cc75'}

# 图1：平行趋势图（全4周）
weekly_all = df12.groupby(['week','treated'])['SalesInThousands'].mean().unstack()
axes[0,0].plot(weekly_all.index, weekly_all[1], marker='o',
               color=colors['treated'], lw=2, label='处理组（方案1）')
axes[0,0].plot(weekly_all.index, weekly_all[0], marker='s',
               color=colors['control'], lw=2, label='对照组（方案2）')
axes[0,0].axvline(2.5, color='gray', linestyle='--', lw=1.5, label='活动前/后分界线')
axes[0,0].fill_between([2.5, 4], 0, 100, alpha=0.05, color='gray')
axes[0,0].set_title('处理组 vs 对照组周度销售趋势（平行趋势检验）')
axes[0,0].set_xlabel('周次')
axes[0,0].set_ylabel('周均销售额（千美元）')
axes[0,0].legend()
axes[0,0].set_xticks([1,2,3,4])
axes[0,0].set_xticklabels(['第1周\n（前）','第2周\n（前）','第3周\n（后）','第4周\n（后）'])

# 图2：2×2 DiD示意图
groups = ['活动前\n（Week1-2）', '活动后\n（Week3-4）']
x = [0, 1]
axes[0,1].plot(x, [pre_t, post_t], 'o-', color=colors['treated'],
               lw=2.5, markersize=10, label=f'处理组（方案1）')
axes[0,1].plot(x, [pre_c, post_c], 's-', color=colors['control'],
               lw=2.5, markersize=10, label=f'对照组（方案2）')
# 反事实线（处理组如果没有处理的预期走势）
counterfactual_post = pre_t + (post_c - pre_c)
axes[0,1].plot([0, 1], [pre_t, counterfactual_post], '--',
               color=colors['treated'], lw=1.5, alpha=0.5, label='反事实趋势')
axes[0,1].annotate(f'DiD = +{did_manual:.2f}K',
                   xy=(1, post_t), xytext=(0.75, (post_t + counterfactual_post)/2),
                   fontsize=11, color='#ee6666',
                   arrowprops=dict(arrowstyle='->', color='#ee6666'))
axes[0,1].set_title(f'双重差分示意图\n（DiD估计量 = +{did_manual:.2f}K）')
axes[0,1].set_ylabel('周均销售额（千美元）')
axes[0,1].set_xticks([0, 1])
axes[0,1].set_xticklabels(groups)
axes[0,1].legend()

# 图3：三种估计方法对比
methods = ['手动DiD', '基础OLS\n回归', '加控制变量\nOLS', '门店固定\n效应']
estimates = [
    did_manual,
    model_base.params['did'],
    model_full.params['did'],
    model_fe.params['did']
]
pvals = [
    stats.ttest_ind(
        df12[(df12['treated']==1)&(df12['post']==1)]['SalesInThousands'],
        df12[(df12['treated']==1)&(df12['post']==0)]['SalesInThousands']
    )[1],
    model_base.pvalues['did'],
    model_full.pvalues['did'],
    model_fe.pvalues['did']
]
bar_colors = ['#5470c6' if p < 0.05 else '#aaaaaa' for p in pvals]
bars = axes[1,0].bar(methods, estimates, color=bar_colors, alpha=0.85, width=0.5)
axes[1,0].axhline(0, color='gray', linestyle='--', lw=1)
for bar, val, p in zip(bars, estimates, pvals):
    sig = '**' if p < 0.05 else 'ns'
    axes[1,0].text(bar.get_x()+bar.get_width()/2,
                   val + 0.2, f'{val:.2f}K\n({sig})',
                   ha='center', fontsize=10)
axes[1,0].set_title('各方法DiD估计量对比')
axes[1,0].set_ylabel('因果效应估计（千美元）')

# 图4：回归系数置信区间图
key_vars = ['treated', 'post', 'did']
coefs = model_full.params[key_vars]
ci    = model_full.conf_int().loc[key_vars]
var_labels = ['treated\n（处理组虚拟）', 'post\n（活动后虚拟）', 'did\n（DiD估计量）']
axes[1,1].errorbar(
    coefs.values, range(len(coefs)),
    xerr=[coefs.values - ci[0].values, ci[1].values - coefs.values],
    fmt='o', color='#5470c6', ecolor='#5470c6', capsize=5,
    markersize=8, elinewidth=2
)
axes[1,1].axvline(0, color='gray', linestyle='--', lw=1)
axes[1,1].set_yticks(range(len(coefs)))
axes[1,1].set_yticklabels(var_labels)
axes[1,1].set_title('OLS回归系数及95%置信区间\n（加控制变量版）')
axes[1,1].set_xlabel('系数估计值（千美元）')

plt.tight_layout()
img_path = OUTPUT_DIR + r'\wa_did_result.png'
plt.savefig(img_path, dpi=150, bbox_inches='tight')
print(f'图表已保存: {img_path}')

# ==========================================
# Step 7：保存汇总结果
# ==========================================
summary = pd.DataFrame({
    '方法': methods,
    'DiD估计量（K$）': [round(e, 4) for e in estimates],
    'p值': [round(p, 4) for p in pvals],
    '显著性': ['**' if p < 0.05 else 'ns' for p in pvals]
})
summary_path = OUTPUT_DIR + r'\did_summary.csv'
summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
print(f'汇总结果已保存: {summary_path}')

print('\n=== DiD分析完成 ===')
print(f'核心结论：方案1相对方案2的因果效应 = +{did_manual:.2f}K/店/周')
print(f'加入控制变量后DiD = +{model_full.params["did"]:.2f}K，'
      f'p = {model_full.pvalues["did"]:.4f}')
