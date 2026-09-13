# ==========================================
# ub_feature_engineering.py
# UserBehavior_2025 特征工程
# Input and output locations are configured through environment variables.
# ==========================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import warnings
import os
warnings.filterwarnings('ignore')

DATA_PATH = os.environ.get('USERBEHAVIOR_DATA_PATH', 'data/raw/UserBehavior_2025.csv')
OUTPUT_DIR = os.environ.get('USERBEHAVIOR_OUTPUT_DIR', 'outputs/userbehavior')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# Step 1：加载数据 + 基础预处理
# ==========================================
print('Step 1: 加载数据...')
df = pd.read_csv(DATA_PATH)
df['datetime'] = pd.to_datetime(df['时间戳'], unit='s')
df['date']     = df['datetime'].dt.date
df['hour']     = df['datetime'].dt.hour
df['month']    = df['datetime'].dt.month
df['weekday']  = df['datetime'].dt.weekday   # 0=周一

print(f'  原始数据: {df.shape[0]:,} 行, {df["用户ID"].nunique():,} 用户')
print(f'  行为分布: {df["行为类型"].value_counts().to_dict()}')

# ==========================================
# Step 2：用户级 RFM 特征
# ==========================================
print('\nStep 2: 构建 RFM 特征...')

snapshot = df['datetime'].max()

# 只用购买行为计算 RFM
buy_df = df[df['行为类型'] == 'buy'].copy()

rfm = buy_df.groupby('用户ID').agg(
    recency   = ('datetime',  lambda x: (snapshot - x.max()).days),
    frequency = ('商品ID',    'count'),
    monetary  = ('售价',      'sum')
).reset_index()

# RFM 分位数打分（1~4分，4最好）
def score_col(series, ascending=True):
    labels = [1, 2, 3, 4] if ascending else [4, 3, 2, 1]
    return pd.qcut(series, q=4, labels=labels, duplicates='drop').astype(float)

rfm['R_score'] = score_col(rfm['recency'],   ascending=False)  # 越近越好
rfm['F_score'] = score_col(rfm['frequency'], ascending=True)
rfm['M_score'] = score_col(rfm['monetary'],  ascending=True)
rfm['RFM_score'] = rfm['R_score'] + rfm['F_score'] + rfm['M_score']

print(f'  RFM 用户数: {len(rfm):,}')
print(f'  RFM 分布:\n{rfm[["recency","frequency","monetary"]].describe().round(2)}')

# ==========================================
# Step 3：用户行为序列特征（转化漏斗）
# ==========================================
print('\nStep 3: 构建行为序列特征...')

behavior_pivot = df.groupby(['用户ID', '行为类型']).size().unstack(fill_value=0)
behavior_pivot.columns = [f'cnt_{c}' for c in behavior_pivot.columns]
behavior_pivot = behavior_pivot.reset_index()

# 补全缺失列
for col in ['cnt_buy', 'cnt_fav', 'cnt_pv']:
    if col not in behavior_pivot.columns:
        behavior_pivot[col] = 0

# 转化率特征
behavior_pivot['pv_to_buy_rate']  = (
    behavior_pivot['cnt_buy'] / (behavior_pivot['cnt_pv'] + 1)
)
behavior_pivot['fav_to_buy_rate'] = (
    behavior_pivot['cnt_buy'] / (behavior_pivot['cnt_fav'] + 1)
)
behavior_pivot['converted'] = (behavior_pivot['cnt_buy'] > 0).astype(int)

print(f'  总体转化率: {behavior_pivot["converted"].mean()*100:.1f}%')

# ==========================================
# Step 4：用户消费偏好特征
# ==========================================
print('\nStep 4: 构建消费偏好特征...')

# 最常购买的品类
fav_category = (df[df['行为类型']=='buy']
                .groupby('用户ID')['商品类别']
                .agg(lambda x: x.value_counts().index[0])
                .reset_index()
                .rename(columns={'商品类别': 'fav_category'}))

# 平均购买单价
avg_price = (buy_df.groupby('用户ID')['售价']
             .mean()
             .reset_index()
             .rename(columns={'售价': 'avg_price'}))

# 品类多样性（购买过几种品类）
category_diversity = (buy_df.groupby('用户ID')['商品类别']
                      .nunique()
                      .reset_index()
                      .rename(columns={'商品类别': 'category_diversity'}))

# ==========================================
# Step 5：时间行为特征
# ==========================================
print('\nStep 5: 构建时间特征...')

time_feat = df.groupby('用户ID').agg(
    active_days   = ('date',    'nunique'),
    peak_hour     = ('hour',    lambda x: x.value_counts().index[0]),
    night_buy_pct = ('hour',    lambda x: ((x >= 22) | (x <= 2)).mean()),
    weekend_pct   = ('weekday', lambda x: (x >= 5).mean())
).reset_index()

# ==========================================
# Step 6：合并所有特征
# ==========================================
print('\nStep 6: 合并特征...')

# 以所有用户为基准（包含无购买行为的用户）
all_users = pd.DataFrame({'用户ID': df['用户ID'].unique()})

feature_df = (all_users
    .merge(rfm,                on='用户ID', how='left')
    .merge(behavior_pivot,     on='用户ID', how='left')
    .merge(fav_category,       on='用户ID', how='left')
    .merge(avg_price,          on='用户ID', how='left')
    .merge(category_diversity, on='用户ID', how='left')
    .merge(time_feat,          on='用户ID', how='left')
)

# 填充未购买用户的缺失值
fill_zero = ['recency','frequency','monetary','R_score','F_score',
             'M_score','RFM_score','cnt_buy','cnt_fav','cnt_pv',
             'pv_to_buy_rate','fav_to_buy_rate','avg_price',
             'category_diversity','active_days','night_buy_pct','weekend_pct']
for col in fill_zero:
    if col in feature_df.columns:
        feature_df[col] = feature_df[col].fillna(0)

feature_df['converted'] = feature_df['converted'].fillna(0).astype(int)
feature_df['peak_hour'] = feature_df['peak_hour'].fillna(12)

# 品类编码
le = LabelEncoder()
feature_df['fav_category'] = feature_df['fav_category'].fillna('未知')
feature_df['fav_category_code'] = le.fit_transform(feature_df['fav_category'])

print(f'  最终特征数据: {feature_df.shape}')
print(f'  转化用户数: {feature_df["converted"].sum():,} '
      f'({feature_df["converted"].mean()*100:.1f}%)')
print(f'\n  特征列表:')
for col in feature_df.columns:
    print(f'    {col}')

# ==========================================
# Step 7：保存特征文件
# ==========================================
feat_path = os.path.join(OUTPUT_DIR, 'user_features.csv')
feature_df.to_csv(feat_path, index=False, encoding='utf-8-sig')
print(f'\n特征文件已保存: {feat_path}')

# ==========================================
# Step 8：可视化（4张图）
# ==========================================
print('\nStep 8: 生成可视化...')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1：RFM 分布（购买用户）
buy_users = feature_df[feature_df['converted'] == 1]
axes[0,0].hist(buy_users['RFM_score'].dropna(), bins=10,
               color='#5470c6', alpha=0.8, edgecolor='white')
axes[0,0].set_title('RFM Score Distribution (buyers)')
axes[0,0].set_xlabel('RFM Score')
axes[0,0].set_ylabel('User Count')

# 图2：各品类购买量
cat_buy = (df[df['行为类型']=='buy']
           .groupby('商品类别').size()
           .sort_values(ascending=True))
axes[0,1].barh(cat_buy.index, cat_buy.values,
               color='#91cc75', alpha=0.85)
axes[0,1].set_title('Purchase Count by Category')
axes[0,1].set_xlabel('Count')

# 图3：转化漏斗
funnel = {
    'pv (浏览)':  df[df['行为类型']=='pv']['用户ID'].nunique(),
    'fav (收藏)': df[df['行为类型']=='fav']['用户ID'].nunique(),
    'buy (购买)': df[df['行为类型']=='buy']['用户ID'].nunique(),
}
axes[1,0].bar(funnel.keys(), funnel.values(),
              color=['#5470c6','#fac858','#91cc75'], alpha=0.85)
for i, (k, v) in enumerate(funnel.items()):
    axes[1,0].text(i, v + 200, f'{v:,}', ha='center', fontsize=11)
axes[1,0].set_title('User Conversion Funnel')
axes[1,0].set_ylabel('Unique Users')

# 图4：月度购买趋势
monthly = (df[df['行为类型']=='buy']
           .groupby('month').size())
axes[1,1].plot(monthly.index, monthly.values,
               marker='o', color='#ee6666', linewidth=2)
axes[1,1].fill_between(monthly.index, monthly.values,
                        alpha=0.1, color='#ee6666')
axes[1,1].set_title('Monthly Purchase Trend')
axes[1,1].set_xlabel('Month')
axes[1,1].set_ylabel('Purchase Count')
axes[1,1].set_xticks(range(1, 13))

plt.tight_layout()
img_path = os.path.join(OUTPUT_DIR, 'ub_feature_analysis.png')
plt.savefig(img_path, dpi=150, bbox_inches='tight')
print(f'图表已保存: {img_path}')

# ==========================================
# Step 9：特征统计摘要
# ==========================================
summary_path = os.path.join(OUTPUT_DIR, 'feature_summary.csv')
feature_df.describe().round(4).to_csv(summary_path, encoding='utf-8-sig')
print(f'统计摘要已保存: {summary_path}')

print('\n=== 特征工程完成 ===')
print(f'输出文件夹: {OUTPUT_DIR}')
print('  - user_features.csv      ← 完整特征矩阵（供模型使用）')
print('  - ub_feature_analysis.png ← 可视化图表')
print('  - feature_summary.csv    ← 特征统计摘要')
