# ==========================================
# log_uplift_model.py
# 日志数据 Uplift 建模（T-Learner + S-Learner）
# Input and output locations are configured through environment variables.
# ==========================================

import json
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ── 字体设置（Windows 中文）─────────────────────────────
matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
matplotlib.rcParams['axes.unicode_minus'] = False

LOG_PATH = os.environ.get('LOG_DATA_PATH', 'data/raw/log_data.json')
OUTPUT_DIR = os.environ.get('LOG_OUTPUT_DIR', 'outputs/log')

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# Step 1：解析 JSON 日志 → 用户级宽表
# ==========================================
print('Step 1: 解析日志数据...')

records = []
with open(LOG_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

print(f'  原始日志条数: {len(records):,}')

rows = []
for r in records:
    common = r.get('common', {})
    uid    = common.get('uid')
    if not uid:
        continue

    # 行为动作
    actions      = r.get('actions', [])
    action_ids   = [a.get('action_id') for a in actions]

    # 曝光展示
    displays     = r.get('displays', [])
    disp_types   = [d.get('display_type') for d in displays]

    # 页面信息
    page         = r.get('page', {})

    rows.append({
        'uid':             uid,
        'ba':              common.get('ba', '未知'),      # 手机品牌
        'os':              common.get('os', '未知'),      # 操作系统
        'ch':              common.get('ch', '未知'),      # 渠道
        'is_new':          int(common.get('is_new', 0)), # 是否新用户
        'vc':              common.get('vc', '未知'),      # App版本
        'has_promotion':   int('promotion' in disp_types),  # 是否曝光促销
        'promotion_cnt':   disp_types.count('promotion'),   # 促销曝光次数
        'has_get_coupon':  int('get_coupon' in action_ids), # 是否领券
        'has_cart_add':    int('cart_add'   in action_ids), # 是否加购
        'has_favor_add':   int('favor_add'  in action_ids), # 是否收藏
        'page_id':         page.get('page_id', ''),
        'during_time':     page.get('during_time', 0),      # 页面停留时长
        'ts':              r.get('ts', 0),
    })

df_raw = pd.DataFrame(rows)
print(f'  展开后行数: {len(df_raw):,}，用户数: {df_raw["uid"].nunique():,}')

# ==========================================
# Step 2：用户级聚合 → 构建特征
# ==========================================
print('\nStep 2: 用户级聚合...')

user_df = df_raw.groupby('uid').agg(
    # treatment：该用户在任意一条日志中被促销曝光 → 实验组
    treatment         = ('has_promotion',  'max'),
    promotion_cnt     = ('promotion_cnt',  'sum'),

    # outcome：领券 or 加购（任一即算转化）
    get_coupon        = ('has_get_coupon', 'max'),
    cart_add          = ('has_cart_add',   'max'),

    # 用户特征
    is_new            = ('is_new',         'max'),
    avg_during_time   = ('during_time',    'mean'),
    log_cnt           = ('uid',            'count'),  # 活跃日志条数

    # 取众数品牌/渠道（用 lambda）
    ba                = ('ba',    lambda x: x.mode()[0] if len(x) else '未知'),
    ch                = ('ch',    lambda x: x.mode()[0] if len(x) else '未知'),
    os                = ('os',    lambda x: x.mode()[0] if len(x) else '未知'),
).reset_index()

# 复合转化标签：领券 OR 加购
user_df['converted'] = ((user_df['get_coupon'] == 1) | (user_df['cart_add'] == 1)).astype(int)

print(f'  用户总数: {len(user_df):,}')
print(f'  实验组（曝光促销）: {user_df["treatment"].sum():,} 人 '
      f'({user_df["treatment"].mean()*100:.1f}%)')
print(f'  对照组（未曝光）: {(user_df["treatment"]==0).sum():,} 人')
print(f'  整体转化率: {user_df["converted"].mean()*100:.1f}%')
print(f'  实验组转化率: {user_df[user_df["treatment"]==1]["converted"].mean()*100:.1f}%')
print(f'  对照组转化率: {user_df[user_df["treatment"]==0]["converted"].mean()*100:.1f}%')

# ==========================================
# Step 3：特征编码
# ==========================================
print('\nStep 3: 特征编码...')

le_ba = LabelEncoder()
le_ch = LabelEncoder()
le_os = LabelEncoder()

user_df['ba_code'] = le_ba.fit_transform(user_df['ba'])
user_df['ch_code'] = le_ch.fit_transform(user_df['ch'])
user_df['os_code'] = le_os.fit_transform(user_df['os'])

FEATURES = ['is_new', 'avg_during_time', 'log_cnt',
            'ba_code', 'ch_code', 'os_code']

X         = user_df[FEATURES].values
T         = user_df['treatment'].values
Y         = user_df['converted'].values

# ==========================================
# Step 4：朴素 ATE 估计（直接对比）
# ==========================================
print('\nStep 4: 朴素 ATE 估计...')

from scipy import stats

y_treated = user_df[user_df['treatment']==1]['converted']
y_control = user_df[user_df['treatment']==0]['converted']

naive_ate = y_treated.mean() - y_control.mean()
t_stat, p_val = stats.ttest_ind(y_treated, y_control)

print(f'  实验组转化率: {y_treated.mean()*100:.2f}%')
print(f'  对照组转化率: {y_control.mean()*100:.2f}%')
print(f'  朴素 ATE: {naive_ate*100:.2f}%')
print(f'  t统计量: {t_stat:.4f}, p值: {p_val:.4f}')
print(f'  结论: {"促销曝光对转化有显著正效果" if p_val < 0.05 else "促销效果在统计上不显著（样本量偏小）"}')

# ==========================================
# Step 5：S-Learner Uplift 模型
# ==========================================
print('\nStep 5: S-Learner...')

# S-Learner：将 treatment 作为一个普通特征放入模型
X_s = np.column_stack([X, T])  # 加入 treatment 列

model_s = GradientBoostingClassifier(
    n_estimators=100, max_depth=3,
    learning_rate=0.05, random_state=42
)
model_s.fit(X_s, Y)

# 预测反事实：同一用户，treatment=1 vs treatment=0
X_t1 = np.column_stack([X, np.ones(len(X))])
X_t0 = np.column_stack([X, np.zeros(len(X))])

uplift_s = (model_s.predict_proba(X_t1)[:, 1] -
            model_s.predict_proba(X_t0)[:, 1])

print(f'  S-Learner 平均 Uplift (CATE均值): {uplift_s.mean()*100:.2f}%')
print(f'  Uplift > 0 的用户比例: {(uplift_s > 0).mean()*100:.1f}%')

# ==========================================
# Step 6：T-Learner Uplift 模型
# ==========================================
print('\nStep 6: T-Learner...')

# T-Learner：分别对实验组和对照组训练独立模型
X_tr = X[T == 1];  Y_tr = Y[T == 1]
X_ct = X[T == 0];  Y_ct = Y[T == 0]

model_t1 = GradientBoostingClassifier(
    n_estimators=100, max_depth=3,
    learning_rate=0.05, random_state=42
)
model_t0 = GradientBoostingClassifier(
    n_estimators=100, max_depth=3,
    learning_rate=0.05, random_state=42
)

model_t1.fit(X_tr, Y_tr)
model_t0.fit(X_ct, Y_ct)

# 对全量用户预测 CATE
uplift_t = (model_t1.predict_proba(X)[:, 1] -
            model_t0.predict_proba(X)[:, 1])

print(f'  T-Learner 平均 Uplift (CATE均值): {uplift_t.mean()*100:.2f}%')
print(f'  Uplift > 0 的用户比例: {(uplift_t > 0).mean()*100:.1f}%')

# ==========================================
# Step 7：用户分层（四象限）
# ==========================================
print('\nStep 7: 用户分层...')

user_df['uplift_t'] = uplift_t
user_df['uplift_s'] = uplift_s

# 使用 T-Learner 结果分四类
median_uplift = np.median(uplift_t)

def classify_user(row):
    high_uplift   = row['uplift_t'] > median_uplift
    high_convert  = row['converted'] == 1
    if high_uplift and high_convert:
        return '可说服用户（重点触达）'
    elif high_uplift and not high_convert:
        return '待激活用户（潜力用户）'
    elif not high_uplift and high_convert:
        return '自然转化用户（无需促销）'
    else:
        return '无效用户（促销无响应）'

user_df['user_type'] = user_df.apply(classify_user, axis=1)
print(user_df['user_type'].value_counts())

# ==========================================
# Step 8：保存结果
# ==========================================
import os
result_path = os.path.join(OUTPUT_DIR, 'log_uplift_result.csv')
user_df[['uid', 'treatment', 'converted', 'uplift_s', 'uplift_t',
         'user_type', 'is_new', 'log_cnt', 'avg_during_time']
       ].sort_values('uplift_t', ascending=False
       ).to_csv(result_path, index=False, encoding='utf-8-sig')
print(f'\n结果已保存: {result_path}')

# ==========================================
# Step 9：可视化（4张图）
# ==========================================
print('\nStep 9: 生成可视化...')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1：实验组 vs 对照组转化率对比
groups   = ['实验组\n（促销曝光）', '对照组\n（未曝光）']
conv_rates = [y_treated.mean()*100, y_control.mean()*100]
bars = axes[0,0].bar(groups, conv_rates, color=['#5470c6','#91cc75'], alpha=0.85, width=0.4)
for bar, val in zip(bars, conv_rates):
    axes[0,0].text(bar.get_x()+bar.get_width()/2, val+0.5,
                   f'{val:.1f}%', ha='center', fontsize=12, fontweight='bold')
axes[0,0].set_title(f'实验组 vs 对照组转化率\n（朴素ATE = {naive_ate*100:.2f}%, p = {p_val:.3f}）')
axes[0,0].set_ylabel('转化率（%）')
axes[0,0].set_ylim(0, max(conv_rates)*1.3)

# 图2：T-Learner Uplift 分布
axes[0,1].hist(uplift_t, bins=30, color='#5470c6', alpha=0.75, edgecolor='white')
axes[0,1].axvline(0, color='red', linestyle='--', lw=1.5, label='零效果线')
axes[0,1].axvline(uplift_t.mean(), color='orange', linestyle='--',
                  lw=1.5, label=f'均值={uplift_t.mean()*100:.1f}%')
axes[0,1].set_title('T-Learner Uplift 分布（个体因果效应CATE）')
axes[0,1].set_xlabel('Uplift 值')
axes[0,1].set_ylabel('用户数量')
axes[0,1].legend()

# 图3：用户分层饼图
type_counts = user_df['user_type'].value_counts()
colors_pie  = ['#5470c6','#91cc75','#fac858','#ee6666']
axes[1,0].pie(type_counts.values, labels=type_counts.index,
              autopct='%1.1f%%', colors=colors_pie,
              startangle=90, pctdistance=0.8)
axes[1,0].set_title('用户四象限分层分布')

# 图4：Uplift 累积增益曲线（Qini Curve 近似）
df_sorted = user_df.sort_values('uplift_t', ascending=False).reset_index(drop=True)
n = len(df_sorted)
cum_treated   = df_sorted['treatment'].cumsum()
cum_converted = df_sorted['converted'].cumsum()

# 计算每个分位的 Uplift 增益
qini_vals = []
for i in range(1, n+1):
    sub = df_sorted.iloc[:i]
    t1  = sub[sub['treatment']==1]['converted']
    t0  = sub[sub['treatment']==0]['converted']
    if len(t0) > 0:
        qini_vals.append(t1.sum() - t0.sum() * len(t1)/len(t0) if len(t1) > 0 else 0)
    else:
        qini_vals.append(0)

axes[1,1].plot(np.arange(1, n+1)/n*100, qini_vals,
               color='#5470c6', lw=2, label='Qini曲线')
axes[1,1].axhline(0, color='gray', linestyle='--', lw=1)
axes[1,1].set_title('Qini 累积增益曲线')
axes[1,1].set_xlabel('触达用户比例（%）')
axes[1,1].set_ylabel('累积增量转化数')
axes[1,1].legend()

plt.tight_layout()
img_path = os.path.join(OUTPUT_DIR, 'log_uplift_result.png')
plt.savefig(img_path, dpi=150, bbox_inches='tight')
print(f'图表已保存: {img_path}')

# ==========================================
# Step 10：输出统计摘要
# ==========================================
print('\n' + '='*50)
print('       日志数据 Uplift 建模结果汇总')
print('='*50)
print(f'  用户总数          : {len(user_df):,}')
print(f'  实验组人数        : {user_df["treatment"].sum():,}')
print(f'  对照组人数        : {(user_df["treatment"]==0).sum():,}')
print(f'  实验组转化率      : {y_treated.mean()*100:.2f}%')
print(f'  对照组转化率      : {y_control.mean()*100:.2f}%')
print(f'  朴素 ATE          : {naive_ate*100:.2f}%  (p={p_val:.4f})')
print(f'  S-Learner CATE均值: {uplift_s.mean()*100:.2f}%')
print(f'  T-Learner CATE均值: {uplift_t.mean()*100:.2f}%')
print(f'  可说服用户数      : {(user_df["user_type"]=="可说服用户（重点触达）").sum():,}')
print('='*50)
print(f'\n输出文件:')
print(f'  {OUTPUT_DIR}\\log_uplift_result.csv   ← 每用户 Uplift 评分')
print(f'  {OUTPUT_DIR}\\log_uplift_result.png   ← 4张可视化图')
