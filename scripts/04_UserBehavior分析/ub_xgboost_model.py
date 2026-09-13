# ==========================================
# ub_xgboost_model.py
# UserBehavior 转化率预测模型
# Input and output locations are configured through environment variables.
# ==========================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, roc_curve,
                             classification_report, confusion_matrix)
import joblib, os, warnings
warnings.filterwarnings('ignore')

FEAT_PATH = os.environ.get('USER_FEATURES_PATH', 'outputs/userbehavior/user_features.csv')
OUTPUT_DIR = os.environ.get('USERBEHAVIOR_OUTPUT_DIR', 'outputs/userbehavior')

# ==========================================
# Step 1：加载特征数据
# ==========================================
print('Step 1: 加载特征数据...')
df = pd.read_csv(FEAT_PATH)

# 模型特征（排除ID、标签、文本列）
FEATURES = [
    'recency', 'frequency', 'monetary',
    'R_score', 'F_score', 'M_score', 'RFM_score',
    'cnt_buy', 'cnt_fav', 'cnt_pv',
    'pv_to_buy_rate', 'fav_to_buy_rate',
    'avg_price', 'category_diversity',
    'active_days', 'peak_hour',
    'night_buy_pct', 'weekend_pct',
    'fav_category_code'
]
LABEL = 'converted'

X = df[FEATURES].values
y = df[LABEL].values

print(f'  样本量: {len(X):,}')
print(f'  特征数: {len(FEATURES)}')
print(f'  转化率: {y.mean()*100:.1f}%  '
      f'(正样本:{y.sum():,} / 负样本:{(1-y).sum():,})')

# ==========================================
# Step 2：划分训练 / 测试集（7:3，分层）
# ==========================================
print('\nStep 2: 划分数据集...')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f'  训练集: {len(X_train):,}  测试集: {len(X_test):,}')

# ==========================================
# Step 3：训练 XGBoost
# ==========================================
print('\nStep 3: 训练 XGBoost...')

# 样本不平衡权重
neg, pos = (y_train==0).sum(), (y_train==1).sum()
scale_pos = neg / pos
print(f'  正负比例: 1:{scale_pos:.2f}，已加权')

model = XGBClassifier(
    n_estimators    = 400,
    max_depth       = 5,
    learning_rate   = 0.05,
    subsample       = 0.8,
    colsample_bytree= 0.8,
    min_child_weight= 5,
    gamma           = 0.1,
    scale_pos_weight= scale_pos,
    eval_metric     = 'auc',
    random_state    = 42,
    n_jobs          = -1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=100
)

# ==========================================
# Step 4：评估指标
# ==========================================
print('\nStep 4: 计算评估指标...')
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

# AUC
auc = roc_auc_score(y_test, y_prob)

# KS
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
ks = max(tpr - fpr)

# Lift（前10%）
df_eval = pd.DataFrame({'y_true': y_test, 'y_prob': y_prob})
df_eval = df_eval.sort_values('y_prob', ascending=False).reset_index(drop=True)
top10 = int(len(df_eval) * 0.1)
lift_10 = df_eval.iloc[:top10]['y_true'].mean() / df_eval['y_true'].mean()

# Lift（前20%）
top20 = int(len(df_eval) * 0.2)
lift_20 = df_eval.iloc[:top20]['y_true'].mean() / df_eval['y_true'].mean()

# PSI（训练集前后半段）
half = len(X_train) // 2
prob_a = model.predict_proba(X_train[:half])[:, 1]
prob_b = model.predict_proba(X_train[half:])[:, 1]
bins = np.percentile(prob_a, np.linspace(0, 100, 11))
bins[0] -= 1e-6; bins[-1] += 1e-6

def calc_psi(exp, act, bins):
    e = np.histogram(exp, bins=bins)[0] / len(exp)
    a = np.histogram(act, bins=bins)[0] / len(act)
    e = np.where(e==0, 1e-6, e)
    a = np.where(a==0, 1e-6, a)
    return np.sum((a - e) * np.log(a / e))

psi = calc_psi(prob_a, prob_b, bins)

# 5折交叉验证 AUC
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_aucs = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)

print('\n' + '='*50)
print('       UserBehavior 模型评估结果')
print('='*50)
print(f'  AUC              : {auc:.4f}')
print(f'  KS               : {ks:.4f}')
print(f'  Lift @ Top 10%   : {lift_10:.2f}x')
print(f'  Lift @ Top 20%   : {lift_20:.2f}x')
print(f'  PSI              : {psi:.4f}  {"(稳定)" if psi<0.1 else "(需关注)"}')
print(f'  5折CV AUC        : {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}')
print('='*50)

print('\n=== 分类报告 ===')
print(classification_report(y_test, y_pred,
      target_names=['未转化','已转化']))

# ==========================================
# Step 5：特征重要性
# ==========================================
feat_imp = pd.DataFrame({
    'feature'   : FEATURES,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print('\n=== 特征重要性 ===')
print(feat_imp.to_string(index=False))

feat_imp.to_csv(
    os.path.join(OUTPUT_DIR, 'feature_importance.csv'),
    index=False, encoding='utf-8-sig'
)

# ==========================================
# Step 6：可视化（4张图）
# ==========================================
print('\nStep 6: 生成可视化...')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1：ROC 曲线
axes[0,0].plot(fpr, tpr, color='#5470c6', lw=2,
               label=f'AUC = {auc:.4f}')
axes[0,0].plot([0,1],[0,1],'k--', lw=1, alpha=0.4)
axes[0,0].fill_between(fpr, tpr, alpha=0.08, color='#5470c6')
axes[0,0].set_title('ROC Curve')
axes[0,0].set_xlabel('False Positive Rate')
axes[0,0].set_ylabel('True Positive Rate')
axes[0,0].legend(loc='lower right')

# 图2：KS 曲线
pop = np.linspace(0, 1, len(tpr))
axes[0,1].plot(pop, tpr, color='#5470c6', lw=2, label='TPR')
axes[0,1].plot(pop, fpr, color='#91cc75', lw=2, label='FPR')
ks_idx = np.argmax(tpr - fpr)
axes[0,1].axvline(pop[ks_idx], color='#ee6666', linestyle='--',
                  lw=1.5, label=f'KS = {ks:.4f}')
axes[0,1].set_title('KS Curve')
axes[0,1].set_xlabel('Population %')
axes[0,1].legend()

# 图3：Lift 曲线（分十档）
deciles = np.arange(0.1, 1.1, 0.1)
lift_vals = [
    df_eval.iloc[:int(len(df_eval)*d)]['y_true'].mean()
    / df_eval['y_true'].mean()
    for d in deciles
]
bars = axes[1,0].bar(range(1, 11), lift_vals,
                     color='#5470c6', alpha=0.8)
axes[1,0].axhline(1.0, color='gray', linestyle='--', lw=1)
for bar, val in zip(bars, lift_vals):
    axes[1,0].text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + 0.02,
                   f'{val:.1f}x', ha='center', fontsize=9)
axes[1,0].set_title('Lift Chart by Decile')
axes[1,0].set_xlabel('Decile (Top N%)')
axes[1,0].set_ylabel('Lift')
axes[1,0].set_xticks(range(1, 11))
axes[1,0].set_xticklabels([f'{int(d*100)}%' for d in deciles])

# 图4：特征重要性（Top 10）
top10_feat = feat_imp.head(10)
axes[1,1].barh(top10_feat['feature'][::-1],
               top10_feat['importance'][::-1],
               color='#91cc75', alpha=0.85)
axes[1,1].set_title('Feature Importance (Top 10)')
axes[1,1].set_xlabel('Importance Score')

plt.tight_layout()
img_path = os.path.join(OUTPUT_DIR, 'ub_model_result.png')
plt.savefig(img_path, dpi=150, bbox_inches='tight')
print(f'图表已保存: {img_path}')

# ==========================================
# Step 7：输出预测结果（带用户ID）
# ==========================================
df_result = df[['用户ID']].copy()
df_result['convert_prob'] = model.predict_proba(X)[:, 1]
df_result['predicted']    = (df_result['convert_prob'] >= 0.5).astype(int)
df_result['actual']       = y
df_result = df_result.sort_values('convert_prob', ascending=False)

result_path = os.path.join(OUTPUT_DIR, 'user_convert_scores.csv')
df_result.to_csv(result_path, index=False, encoding='utf-8-sig')
print(f'用户评分已保存: {result_path}')

# ==========================================
# Step 8：保存模型
# ==========================================
model_path = os.path.join(OUTPUT_DIR, 'xgb_ub_model.pkl')
joblib.dump(model, model_path)
print(f'模型已保存: {model_path}')

print('\n=== 完成 ===')
print(f'输出文件夹: {OUTPUT_DIR}')
print('  - ub_model_result.png       ← 4张评估图')
print('  - feature_importance.csv    ← 特征重要性')
print('  - user_convert_scores.csv   ← 每位用户转化概率')
print('  - xgb_ub_model.pkl          ← 训练好的模型')
