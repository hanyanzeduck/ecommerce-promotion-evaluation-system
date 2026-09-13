# ==========================================
# criteo_conversion_model.py
# 用175个分片训练转化率预测模型
# 策略：抽样合并 → XGBoost → AUC/KS/Lift
# ==========================================

import pandas as pd
import numpy as np
import glob
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

DATA_DIR = os.environ.get('CRITEO_DATA_DIR', 'data/raw/criteo_split')
FEATURES = [f'f{i}' for i in range(12)] # f0~f11
LABEL = 'conversion'

# ==========================================
# Step 1：分块抽样读取（每片抽20%，控制内存）
# 175片 × 8万行 × 20% ≈ 280万行，约400MB，安全
# ==========================================
print('正在读取数据...')
chunks = []

for i in range(1, 176):
    fpath = os.path.join(DATA_DIR, f'criteo_{i:03d}.csv')
    try:
        df = pd.read_csv(fpath)
        if LABEL not in df.columns:
            print(f' 跳过 {i}: 无 {LABEL} 列')
            continue
        
        # 提取需要的列
        cols = FEATURES + [LABEL]
        df = df[cols].copy()
        
        # 分层抽样：保持正负样本比例
        if df[LABEL].nunique() > 1:  # 确保有多个类别
            # 简单随机抽样代替分层抽样
            df_sample = df.sample(frac=0.2, random_state=42)
        else:
            df_sample = df.sample(frac=0.2, random_state=42)
        
        chunks.append(df_sample)
        if i % 20 == 0:
            print(f' 已读取 {i}/175 个分片...')
    except Exception as e:
        print(f' 跳过 {i}: {e}')

print(f' 成功读取 {len(chunks)} 个分片')
df_all = pd.concat(chunks, ignore_index=True)
print(f'合并后列名: {df_all.columns.tolist()}')

df_all = pd.concat(chunks, ignore_index=True)
print(f'\n合并后总行数: {len(df_all):,}')
print(f'转化率: {df_all[LABEL].mean()*100:.3f}%')
print(f'转化样本数: {df_all[LABEL].sum():,}')

# ==========================================
# Step 2：划分训练集 / 测试集（8:2）
# ==========================================
X = df_all[FEATURES].values
y = df_all[LABEL].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f'\n训练集: {len(X_train):,} 行')
print(f'测试集: {len(X_test):,} 行')

# ==========================================
# Step 3：训练 XGBoost
# ==========================================
print('\n开始训练 XGBoost...')

# 正负样本不平衡处理
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos = neg / pos
print(f'正负样本比: 1:{scale_pos:.0f}，已自动加权')

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos, # 处理样本不平衡
    eval_metric='auc',
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)

# ==========================================
# Step 4：评估指标
# ==========================================
y_prob = model.predict_proba(X_test)[:, 1]

# AUC
auc = roc_auc_score(y_test, y_prob)

# KS 统计量
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
ks = max(tpr - fpr)

# Lift（前10%用户的提升度）
df_eval = pd.DataFrame({'y_true': y_test, 'y_prob': y_prob})
df_eval = df_eval.sort_values('y_prob', ascending=False).reset_index(drop=True)
top10_pct = int(len(df_eval) * 0.1)
lift_10 = (df_eval.iloc[:top10_pct]['y_true'].mean()) / df_eval['y_true'].mean()

# PSI（用训练集前半 vs 后半分片模拟跨场景稳定性）
half = len(X_train) // 2
prob_a = model.predict_proba(X_train[:half])[:, 1]
prob_b = model.predict_proba(X_train[half:])[:, 1]
bins = np.percentile(prob_a, np.linspace(0, 100, 11))
bins[0] -= 1e-6; bins[-1] += 1e-6

def calc_psi(expected, actual, bins):
    exp_pct = np.histogram(expected, bins=bins)[0] / len(expected)
    act_pct = np.histogram(actual, bins=bins)[0] / len(actual)
    exp_pct = np.where(exp_pct == 0, 1e-6, exp_pct)
    act_pct = np.where(act_pct == 0, 1e-6, act_pct)
    return np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))

psi = calc_psi(prob_a, prob_b, bins)

print('\n' + '='*45)
print(' 模型评估结果')
print('='*45)
print(f' AUC : {auc:.4f}')
print(f' KS : {ks:.4f}')
print(f' Lift@Top10% : {lift_10:.2f}x')
print(f' PSI : {psi:.4f} {"(稳定)" if psi < 0.1 else "(需关注)"}')
print('='*45)

# ==========================================
# Step 5：特征重要性
# ==========================================
feat_imp = pd.DataFrame({
    'feature': FEATURES,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print('\n=== 特征重要性 Top 12 ===')
print(feat_imp.to_string(index=False))

# ==========================================
# Step 6：可视化（4张图）
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1：ROC 曲线
axes[0,0].plot(fpr, tpr, color='#5470c6', lw=2,
               label=f'AUC = {auc:.4f}')
axes[0,0].plot([0,1],[0,1],'k--', lw=1, alpha=0.5)
axes[0,0].fill_between(fpr, tpr, alpha=0.08, color='#5470c6')
axes[0,0].set_title('ROC Curve')
axes[0,0].set_xlabel('False Positive Rate')
axes[0,0].set_ylabel('True Positive Rate')
axes[0,0].legend()

# 图2：KS 曲线
axes[0,1].plot(np.linspace(0,1,len(tpr)), tpr,
              color='#5470c6', lw=2, label='TPR')
axes[0,1].plot(np.linspace(0,1,len(fpr)), fpr,
              color='#91cc75', lw=2, label='FPR')
ks_idx = np.argmax(tpr - fpr)
axes[0,1].axvline(ks_idx/len(tpr), color='#ee6666',
                  linestyle='--', lw=1.5,
                  label=f'KS = {ks:.4f}')
axes[0,1].set_title('KS Curve')
axes[0,1].set_xlabel('Population %')
axes[0,1].legend()

# 图3：Lift 曲线
deciles = np.arange(0.1, 1.1, 0.1)
lift_vals = []
for d in deciles:
    n = int(len(df_eval) * d)
    lift_vals.append(
        df_eval.iloc[:n]['y_true'].mean() / df_eval['y_true'].mean()
    )
axes[1,0].bar(range(1,11), lift_vals, color='#5470c6', alpha=0.8)
axes[1,0].axhline(1.0, color='gray', linestyle='--', lw=1)
axes[1,0].set_title('Lift Chart by Decile')
axes[1,0].set_xlabel('Decile (Top N%)')
axes[1,0].set_ylabel('Lift')
axes[1,0].set_xticks(range(1,11))
axes[1,0].set_xticklabels([f'{int(d*100)}%' for d in deciles])

# 图4：特征重要性
axes[1,1].barh(feat_imp['feature'][::-1],
               feat_imp['importance'][::-1],
               color='#91cc75', alpha=0.85)
axes[1,1].set_title('Feature Importance')
axes[1,1].set_xlabel('Importance Score')

plt.tight_layout()
os.makedirs(os.environ.get('CRITEO_OUTPUT_DIR', 'outputs/criteo'), exist_ok=True)
output = os.path.join(os.environ.get('CRITEO_OUTPUT_DIR', 'outputs/criteo'), 'criteo_model_result.png')
plt.savefig(output, dpi=150, bbox_inches='tight')
print(f'\n图表已保存: {output}')

# ==========================================
# Step 7：保存模型
# ==========================================
import joblib
model_path = os.path.join(os.environ.get('CRITEO_OUTPUT_DIR', 'outputs/criteo'), 'xgb_conversion_model.pkl')
joblib.dump(model, model_path)
print(f'模型已保存: {model_path}')
print('\n=== 完成 ===')
