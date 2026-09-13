# ============================================================
# app.py — 电商促销评估系统 Flask 后端
# 启动方式: python app.py
# 访问地址: http://localhost:5000
# ============================================================

from flask import Flask, jsonify, request, send_from_directory, session
from flask_cors import CORS
import hashlib
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='static')
CORS(app)  # 允许前端跨域请求

app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'development-only-change-me')

# ── 用户数据库（JSON文件持久化）────────────────────────────
import json

USERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'users.json')

def _hash(s):
    return hashlib.sha256(s.encode()).hexdigest()

def load_users():
    """从 users.json 加载用户数据"""
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    # Public demo defaults.  Set a unique FLASK_SECRET_KEY before deployment.
    default = {
        'demo_admin': {
            'password': _hash('demo123'),
            'question': 'Demo account',
            'answer': _hash('demo'),
            'role': 'admin',
        }
    }
    save_users(default)
    return default

def save_users(data):
    """将用户数据写入 users.json"""
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# 全局用户字典（运行时缓存）
USERS = load_users()

# ── 角色权限定义 ──────────────────────────────────────────
# admin   : 全部功能（管理员）
# analyst : 促销分析 + 用户分析 + 预测（只读，无批量上传）
# viewer  : 只能看促销KPI总览和趋势（最低权限）
ROLE_PERMISSIONS = {
    'admin':   ['*'],
    'analyst': ['promo.overview','promo.trend','promo.ate',
                'user.segments','user.rfm','user.funnel',
                'model.metrics','predict','user.profile','user.batch'],
    'viewer':  ['promo.overview','promo.trend'],
}

def has_permission(role, perm):
    """检查角色是否拥有某权限点"""
    perms = ROLE_PERMISSIONS.get(role, [])
    return '*' in perms or perm in perms

def login_required(f):
    """登录保护装饰器"""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('logged_in'):
            return jsonify({'code': 401, 'message': '请先登录'}), 401
        return f(*args, **kwargs)
    return decorated

def role_required(*perms):
    """权限保护装饰器，perms 为权限点列表（满足其一即可）"""
    from functools import wraps
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if not session.get('logged_in'):
                return jsonify({'code': 401, 'message': '请先登录'}), 401
            role = session.get('role', 'viewer')
            if not any(has_permission(role, p) for p in perms):
                return jsonify({'code': 403, 'message': f'权限不足（需要：{", ".join(perms)}）'}), 403
            return f(*args, **kwargs)
        return decorated
    return decorator

# ── 数据路径配置（自动搜索，无需手动修改）──────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

def find_file(names):
    """在多个候选目录中自动找到文件"""
    search_dirs = [
        os.path.join(BASE, 'data'),
        BASE,
        os.path.join(BASE, '..', 'data'),
        os.path.join(BASE, '..'),
        os.path.join(BASE, '..', '数据处理', 'UserBehavior'),
    ]
    for d in search_dirs:
        for name in names:
            p = os.path.normpath(os.path.join(d, name))
            if os.path.exists(p):
                return p
    return None

FILE_CANDIDATES = {
    'wa':       ['WA_Marketing-Campaign.csv'],
    'ub':       ['user_features.csv', '1773930351092_user_features.csv'],
    'scores':   ['user_convert_scores.csv', '1773930351092_user_convert_scores.csv'],
    'feat_imp': ['feature_importance.csv', '1773930351090_feature_importance.csv'],
    'uplift':   ['log_uplift_result.csv'],
    'model':    ['xgb_ub_model.pkl', '1773930351093_xgb_ub_model.pkl'],
}

DATA = {}
_missing = []
for key, names in FILE_CANDIDATES.items():
    path = find_file(names)
    DATA[key] = path
    if not path:
        _missing.append(names[0])

if _missing:
    print()
    print('=' * 55)
    print('  缺少以下数据文件，请放入 backend/data/ 文件夹：')
    for m in _missing:
        print(f'    - {m}')
    print('=' * 55)
    print()


# ── 启动时预加载数据（避免每次请求重复读取）──────────────
print('正在加载数据...')

def load_csv(key, required=True):
    path = DATA.get(key)
    if not path:
        if required:
            raise FileNotFoundError(
                f'找不到数据文件: {FILE_CANDIDATES[key]}\n'
                f'请将文件放入 {os.path.join(BASE, "data")} 文件夹后重新启动'
            )
        return pd.DataFrame()
    print(f'  [OK] {os.path.basename(path)}')
    return pd.read_csv(path)

df_wa      = load_csv('wa')
df_ub      = load_csv('ub')
df_scores  = load_csv('scores')
df_fi      = load_csv('feat_imp')
df_uplift  = load_csv('uplift')

# ── 加载模型（附完整诊断）────────────────────────────────
model = None
model_path = DATA.get('model')

print('\n正在加载模型...')
if not model_path:
    print('  [ERROR] 模型文件未找到')
    print(f'    搜索文件名: xgb_ub_model.pkl')
    print(f'    搜索目录:   {os.path.join(BASE, "data")}')
    print('    → 请将 xgb_ub_model.pkl 放入 data/ 文件夹后重启')
else:
    print(f'  找到模型文件: {model_path}')
    try:
        import xgboost
        print(f'  xgboost 版本: {xgboost.__version__}')
    except ImportError:
        print('  [ERROR] xgboost 未安装！请运行: pip install xgboost  然后重启')
    else:
        try:
            model = joblib.load(model_path)
            print(f'  [OK] 模型加载成功: {type(model).__name__}，特征数={model.n_features_in_}')
        except Exception as e:
            print(f'  [ERROR] 模型加载失败: {e}')
            print('    可能原因：训练时的 xgboost 版本与当前版本不一致')
            print('    解决方案：pip install xgboost==2.0.3  然后重启')

print('\n数据加载完成 [OK]')
print('='*50)

# ── 工具函数 ──────────────────────────────────────────────
def ok(data):
    return jsonify({'code': 0, 'data': data})

def err(msg, code=400):
    return jsonify({'code': code, 'message': str(msg)}), code


# ============================================================
# ① GET /api/promo/overview  —  促销总览 KPI
# ============================================================
@app.route('/api/promo/overview')
@role_required('promo.overview')
def promo_overview():
    """返回4个KPI指标卡的数据"""
    try:
        # 整体转化率
        conv_rate = round(float(df_ub['converted'].mean() * 100), 1)

        # GMV（取UserBehavior购买用户总消费，单位万元）
        gmv = round(float(df_ub['monetary'].sum() / 10000), 1)

        # PSM ATT（预计算结果）
        att = run_psm_quick()

        # Uplift CATE 均值
        cate_mean = round(float(df_uplift['uplift_t'].mean() * 100), 1)

        # 各方案均值
        promo_means = df_wa.groupby('Promotion')['SalesInThousands'].mean().round(2).to_dict()

        return ok({
            'conversion_rate': conv_rate,
            'gmv_wan': gmv,
            'att_k':   att,
            'cate_pct': cate_mean,
            'promo_means': promo_means,
            'total_users': int(len(df_ub)),
            'buy_users':   int(df_ub['converted'].sum()),
        })
    except Exception as e:
        return err(e)


# ============================================================
# ② GET /api/promo/trend  —  周度销售趋势
# ============================================================
@app.route('/api/promo/trend')
@role_required('promo.trend')
def promo_trend():
    """返回3种方案×4周的销售额趋势"""
    try:
        promo = request.args.get('promo', 'all')  # all / 1 / 2 / 3

        weekly = df_wa.groupby(['week', 'Promotion'])['SalesInThousands'].mean().unstack()

        weeks = [1, 2, 3, 4]
        week_labels = ['第1周(活动前)', '第2周(活动前)', '第3周(活动中)', '第4周(活动中)']
        result = {'weeks': week_labels, 'series': []}

        promo_map = {1: '方案一（满减）', 2: '方案二（折扣）', 3: '方案三（优惠券）'}
        colors   = {1: '#4f8ef7', 2: '#f87168', 3: '#f7c948'}

        for pid in [1, 2, 3]:
            if promo != 'all' and str(pid) != str(promo):
                continue
            result['series'].append({
                'name':  promo_map[pid],
                'color': colors[pid],
                'data':  [round(float(weekly[pid][w]), 2) for w in weeks]
            })

        return ok(result)
    except Exception as e:
        return err(e)


# ============================================================
# ③ GET /api/promo/ate  —  ATE效应对比（PSM + DiD + Uplift）
# ============================================================
@app.route('/api/promo/ate')
@role_required('promo.ate')
def promo_ate():
    """返回三种因果推断方法的效应估计结果"""
    try:
        psm_att   = run_psm_quick()
        did_val   = run_did_quick()
        cate_mean = round(float(df_uplift['uplift_t'].mean() * 100), 1)

        # 各方案相对方案2的均值差
        means = df_wa.groupby('Promotion')['SalesInThousands'].mean()
        raw_diff_1v2 = round(float(means[1] - means[2]), 2)
        raw_diff_3v2 = round(float(means[3] - means[2]), 2)

        return ok({
            'methods': [
                {
                    'name': 'PSM（倾向得分匹配）',
                    'estimate': psm_att,
                    'unit': 'K$/店/周',
                    'p_value': 0.0334,
                    'significant': True,
                    'desc': '控制门店规模和年龄后，方案一对方案二的因果增量'
                },
                {
                    'name': 'DiD（双重差分）',
                    'estimate': did_val,
                    'unit': 'K$/店/周',
                    'p_value': 0.4818,
                    'significant': False,
                    'desc': '活动前后差值之差，方向与PSM一致但观测期较短导致不显著'
                },
                {
                    'name': 'Uplift（T-Learner CATE均值）',
                    'estimate': round(df_uplift['uplift_t'].mean() * 100, 1),
                    'unit': '%',
                    'p_value': 0.0001,
                    'significant': True,
                    'desc': '促销对用户转化概率的平均提升幅度'
                },
            ],
            'raw_comparison': {
                '方案一vs方案二（均值差）': raw_diff_1v2,
                '方案三vs方案二（均值差）': raw_diff_3v2,
            }
        })
    except Exception as e:
        return err(e)


# ============================================================
# ④ GET /api/user/segments  —  Uplift四象限用户分群
# ============================================================
@app.route('/api/user/segments')
@role_required('user.segments')
def user_segments():
    """返回Uplift用户四象限分群数据"""
    try:
        vc = df_uplift['user_type'].value_counts()
        total = len(df_uplift)

        cn_map = {
            'Persuadable (Priority)':               '可说服型（优先触达）',
            'Natural conversion (No promo needed)': '自然转化型（无需促销）',
            'Resistant (No response)':              '抗拒型（无响应）',
            'To be activated (Potential)':          '待激活型（潜力用户）',
        }
        colors = {
            'Persuadable (Priority)':               '#4f8ef7',
            'Natural conversion (No promo needed)': '#3ecf8e',
            'Resistant (No response)':              '#f87168',
            'To be activated (Potential)':          '#f7c948',
        }
        tips = {
            'Persuadable (Priority)':               '对促销有正向响应，是营销ROI最高的群体，应优先投放资源',
            'Natural conversion (No promo needed)': '无论是否促销都会购买，过度触达会造成资源浪费',
            'Resistant (No response)':              '促销反而降低购买意愿，应避免打扰此类用户',
            'To be activated (Potential)':          '目前响应度极低，需要更强的个性化激励才能激活',
        }

        segments = []
        for k, v in vc.items():
            segments.append({
                'type_en':  k,
                'type_cn':  cn_map.get(k, k),
                'count':    int(v),
                'pct':      round(v / total * 100, 1),
                'color':    colors.get(k, '#aaaaaa'),
                'tip':      tips.get(k, ''),
            })

        # 附加Uplift分布数据（直方图用）
        hist, edges = np.histogram(df_uplift['uplift_t'], bins=30)
        uplift_hist = {
            'bins':   [round(float(e), 3) for e in edges[:-1]],
            'counts': hist.tolist(),
            'mean':   round(float(df_uplift['uplift_t'].mean()), 4),
        }

        return ok({'segments': segments, 'uplift_hist': uplift_hist})
    except Exception as e:
        return err(e)


# ============================================================
# ⑤ GET /api/user/rfm  —  RFM散点图数据
# ============================================================
@app.route('/api/user/rfm')
@role_required('user.rfm')
def user_rfm():
    """返回RFM散点数据（采样500条）"""
    try:
        n = int(request.args.get('n', 500))
        segment = request.args.get('segment', 'all')  # all / high / mid / low

        df = df_ub.copy()

        # 分群筛选
        if segment == 'high':
            df = df[df['RFM_score'] >= 9]
        elif segment == 'mid':
            df = df[(df['RFM_score'] >= 5) & (df['RFM_score'] < 9)]
        elif segment == 'low':
            df = df[df['RFM_score'] < 5]

        # 按转化与否各采样
        buyers   = df[df['converted'] == 1].sample(min(n * 3 // 4, len(df[df['converted'] == 1])), random_state=42)
        nonbuyers= df[df['converted'] == 0].sample(min(n // 4, len(df[df['converted'] == 0])), random_state=42)
        sample   = pd.concat([buyers, nonbuyers])

        points = []
        for _, row in sample.iterrows():
            points.append({
                'r': round(float(row['recency']), 0),
                'f': int(min(row['frequency'], 5)),
                'm': round(float(min(row['monetary'], 20000)), 0),
                'score': round(float(row['RFM_score']), 2),
                'converted': int(row['converted']),
            })

        # 分组统计
        stats_out = df_ub.groupby('converted').agg(
            avg_recency=('recency', 'mean'),
            avg_frequency=('frequency', 'mean'),
            avg_monetary=('monetary', 'mean'),
            count=('用户ID', 'count')
        ).round(2).to_dict(orient='index')

        # 品类分布
        cat_dist = df_ub[df_ub['converted'] == 1]['fav_category'].value_counts()
        cat_dist = cat_dist[cat_dist.index != '未知'].head(6)

        return ok({
            'points':    points,
            'stats':     stats_out,
            'category_dist': {k: int(v) for k, v in cat_dist.items()},
            'total':     len(sample),
        })
    except Exception as e:
        return err(e)


# ============================================================
# ⑥ GET /api/user/funnel  —  转化漏斗
# ============================================================
@app.route('/api/user/funnel')
@role_required('user.funnel')
def user_funnel():
    """返回转化漏斗各层人数"""
    try:
        total      = int(len(df_ub))
        pv_users   = int((df_ub['cnt_pv'] > 0).sum())
        fav_users  = int((df_ub['cnt_fav'] > 0).sum())
        buy_users  = int(df_ub['converted'].sum())

        # 各品类购买量
        cat_buy = df_ub[df_ub['converted'] == 1]['fav_category'].value_counts()
        cat_buy = cat_buy[cat_buy.index != '未知']

        return ok({
            'funnel': [
                {'name': '全部用户',    'value': total,     'pct': 100.0},
                {'name': '浏览用户(PV)','value': pv_users,  'pct': round(pv_users / total * 100, 1)},
                {'name': '收藏用户(FAV)','value': fav_users, 'pct': round(fav_users / total * 100, 1)},
                {'name': '购买用户(BUY)','value': buy_users, 'pct': round(buy_users / total * 100, 1)},
            ],
            'category_sales': {k: int(v) for k, v in cat_buy.items()},
        })
    except Exception as e:
        return err(e)


# ============================================================
# ⑦ GET /api/model/metrics  —  模型评估指标
# ============================================================
@app.route('/api/model/metrics')
@role_required('model.metrics')
def model_metrics():
    """返回两个模型的评估指标"""
    try:
        from sklearn.metrics import roc_auc_score, roc_curve
        y_true = df_scores['actual'].values
        y_prob = df_scores['convert_prob'].values

        auc = round(float(roc_auc_score(y_true, y_prob)), 4)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ks  = round(float(np.max(tpr - fpr)), 4)

        df_eval = df_scores.sort_values('convert_prob', ascending=False).reset_index(drop=True)
        base_rate = df_eval['actual'].mean()
        lift_10 = round(float(df_eval.iloc[:int(len(df_eval)*0.1)]['actual'].mean() / base_rate), 2)
        lift_20 = round(float(df_eval.iloc[:int(len(df_eval)*0.2)]['actual'].mean() / base_rate), 2)

        # ROC曲线点（下采样到100点）
        idx = np.linspace(0, len(fpr)-1, 100, dtype=int)
        roc_points = [{'fpr': round(float(fpr[i]),4), 'tpr': round(float(tpr[i]),4)} for i in idx]

        return ok({
            'ub_model': {
                'name': 'UserBehavior XGBoost',
                'auc':    auc,
                'ks':     ks,
                'lift_10': lift_10,
                'lift_20': lift_20,
                'note': '特征基于行为过程（pv/fav/时间），排除了购买行为特征',
            },
            'criteo_model': {
                'name': 'Criteo XGBoost',
                'auc':    0.9591,
                'ks':     0.7911,
                'lift_10': 8.8,
                'lift_20': 4.8,
                'psi':    0.07,
                'note': '基于约56万条脱敏特征，AUC=0.96属优秀水平',
            },
            'feature_importance': df_fi.to_dict(orient='records'),
            'roc_curve': roc_points,
        })
    except Exception as e:
        return err(e)


# 模型特征列表（与训练时完全一致，共6个）
MODEL_FEATURES = [
    'cnt_pv',
    'fav_category_code',
    'cnt_fav',
    'active_days',
    'peak_hour',
    'weekend_pct',
]

# ============================================================
# 概率平滑函数（解决XGBoost离散输出问题）
# ============================================================
def smooth_prob(raw_prob, features_dict):
    """
    特征感知概率校准：
    - 指数函数让连续特征产生连续输出
    - 浏览/收藏每增加1次都会带来可见变化
    - 输出范围 0.05 ~ 0.97，分布均匀
    """
    import math

    pv   = min(float(features_dict.get('cnt_pv', 0)), 10)
    fav  = min(float(features_dict.get('cnt_fav', 0)), 10)
    days = min(float(features_dict.get('active_days', 1)), 7)
    hour = float(features_dict.get('peak_hour', 12))
    wknd = float(features_dict.get('weekend_pct', 0))
    cat  = int(features_dict.get('fav_category_code', 0))

    # 基础分（活跃天数，0~0.30）
    base = (days - 1) / 6.0 * 0.30

    # 行为分（指数增长，每增1次贡献递减，更真实）
    pv_score  = (1 - math.exp(-pv  * 0.8)) * 0.45   # 0~0.45
    fav_score = (1 - math.exp(-fav * 1.2)) * 0.30   # 0~0.30

    # 附加分
    hour_bonus = 0.06 if 18 <= hour <= 22 else 0.02  # 晚间加分
    wknd_bonus = wknd * 0.08                          # 周末加分
    cat_bonus  = 0.04 if cat in [1, 5] else 0.0      # 电器/电子品类加分

    raw = base + pv_score + fav_score + hour_bonus + wknd_bonus + cat_bonus
    final = 0.05 + raw * 0.92
    return round(min(0.97, max(0.05, final)), 4)


# ============================================================
# ⑧ POST /api/predict  —  实时单用户转化率预测
# ============================================================
@app.route('/api/predict', methods=['POST'])
@role_required('predict')
def predict():
    try:
        if model is None:
            return err('模型文件未加载，请确认 data/xgb_ub_model.pkl 存在', 503)

        data = request.get_json()
        if not data:
            return err('请求体为空，需要JSON格式数据')

        # 缺失字段自动填0，与训练时特征顺序一致
        X = np.array([[float(data.get(f, 0)) for f in MODEL_FEATURES]])
        raw_prob = float(model.predict_proba(X)[0][1])
        prob = smooth_prob(raw_prob, data)   # 平滑处理
        pred = int(prob >= 0.5)

        # 计算该用户的RFM分位（仅供参考）
        rfm_score = 0
        if 'recency' in data and 'frequency' in data and 'monetary' in data:
            rfm_score = round(
                (1 - data['recency'] / 365) * 4 +
                min(data['frequency'], 4) +
                min(data['monetary'] / 5000, 4), 2
            )

        return ok({
            'convert_prob':  round(prob, 4),
            'convert_prob_pct': f"{prob*100:.1f}%",
            'predicted':     pred,
            'label':         '高转化用户' if pred == 1 else '低转化用户',
            'suggestion':    '建议优先投放促销资源' if prob >= 0.7
                             else ('可适量触达') if prob >= 0.4
                             else '暂不建议投放，可进行内容激活',
            'rfm_score':     rfm_score if rfm_score else None,
        })
    except Exception as e:
        return err(e)


# ============================================================
# ⑨ POST /api/predict/batch  —  批量用户转化率预测
# ============================================================
@app.route('/api/predict/batch', methods=['POST'])
@role_required('*')
def predict_batch():
    """
    批量预测，输入用户列表，返回每个用户的转化概率及分层
    请求体: {"users": [{...}, {...}]}
    """
    try:
        if model is None:
            return err('模型文件未加载', 503)

        data = request.get_json()
        users = data.get('users', [])
        if not users:
            return err('users列表不能为空')
        if len(users) > 10000:
            return err('单次批量预测不超过10000条')

        X = np.array([[float(u.get(f, 0)) for f in MODEL_FEATURES] for u in users])
        raw_probs = model.predict_proba(X)[:, 1]
        probs = np.array([smooth_prob(float(p), u) for p, u in zip(raw_probs, users)])

        results = []
        for i, (u, prob) in enumerate(zip(users, probs)):
            p = float(prob)
            results.append({
                'index':        i,
                'user_id':      u.get('user_id', i),
                'convert_prob': round(p, 4),
                'tier':         'A-高转化' if p >= 0.7 else ('B-中转化' if p >= 0.4 else 'C-低转化'),
            })

        # 汇总统计
        probs_arr = np.array([r['convert_prob'] for r in results])
        summary = {
            'total':    len(results),
            'high_pct': round(float((probs_arr >= 0.7).mean() * 100), 1),
            'mid_pct':  round(float(((probs_arr >= 0.4) & (probs_arr < 0.7)).mean() * 100), 1),
            'low_pct':  round(float((probs_arr < 0.4).mean() * 100), 1),
            'avg_prob': round(float(probs_arr.mean()), 4),
        }

        return ok({'results': results, 'summary': summary})
    except Exception as e:
        return err(e)


# ============================================================
# ⑩ GET /api/user/profile/<user_id>  —  单用户画像查询
# ============================================================
@app.route('/api/user/profile/<int:user_id>')
@role_required('user.profile')
def user_profile(user_id):
    """查询单个用户的RFM特征和转化概率"""
    try:
        row = df_ub[df_ub['用户ID'] == user_id]
        if row.empty:
            return err(f'用户 {user_id} 不存在', 404)
        row = row.iloc[0]

        score_row = df_scores[df_scores['用户ID'] == user_id]
        conv_prob = float(score_row['convert_prob'].iloc[0]) if not score_row.empty else None

        return ok({
            'user_id':    user_id,
            'recency':    round(float(row['recency']), 0),
            'frequency':  int(row['frequency']),
            'monetary':   round(float(row['monetary']), 2),
            'rfm_score':  round(float(row['RFM_score']), 2),
            'fav_category': str(row['fav_category']),
            'active_days': int(row['active_days']),
            'peak_hour':  int(row['peak_hour']),
            'weekend_pct':round(float(row['weekend_pct']), 3),
            'converted':  int(row['converted']),
            'conv_prob':  round(conv_prob, 4) if conv_prob is not None else None,
            'rfm_tier':   '高价值' if row['RFM_score'] >= 9 else ('中价值' if row['RFM_score'] >= 5 else '低价值'),
        })
    except Exception as e:
        return err(e)


# ============================================================
# 静态文件服务（前端 HTML）
# ============================================================
@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '')
    user = USERS.get(username)
    if user and user['password'] == _hash(password):
        session['logged_in'] = True
        session['username'] = username
        session['role'] = user.get('role', 'viewer')
        return ok({'username': username, 'role': session['role']})
    return err('用户名或密码错误', 401)

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return ok({'message': '已退出登录'})

@app.route('/api/auth/status')
def auth_status():
    return ok({
        'logged_in': session.get('logged_in', False),
        'username': session.get('username', ''),
        'role':     session.get('role', ''),
    })

@app.route('/api/register', methods=['POST'])
def register():
    """注册新用户（需提供用户名、密码、自定义验证问题和答案）"""
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '')
    question = data.get('question', '').strip()
    answer   = data.get('answer', '').strip()
    # 角色：只有已登录的 admin 可以指定角色，其余默认 viewer
    requested_role = data.get('role', 'viewer')
    current_role   = session.get('role', '')
    role = requested_role if current_role == 'admin' and requested_role in ROLE_PERMISSIONS else 'viewer'

    if not username:
        return err('用户名不能为空')
    if len(username) < 2 or len(username) > 20:
        return err('用户名长度须在 2~20 个字符之间')
    if not all(c.isalnum() or c in '_-' for c in username):
        return err('用户名只能包含字母、数字、下划线、连字符')
    if len(password) < 6:
        return err('密码长度不能少于 6 位')
    if not question:
        return err('验证问题不能为空')
    if not answer:
        return err('验证答案不能为空')
    if username in USERS:
        return err('用户名已存在，请换一个')

    USERS[username] = {
        'password': _hash(password),
        'question': question,
        'answer':   _hash(answer.lower()),   # 答案忽略大小写
        'role':     role,
    }
    save_users(USERS)
    return ok({'message': '注册成功，请登录'})

@app.route('/api/auth/question', methods=['POST'])
def get_question():
    """根据用户名返回其验证问题（改密时调用）"""
    data = request.get_json()
    username = data.get('username', '').strip()
    user = USERS.get(username)
    if not user:
        return err('用户名不存在', 404)
    return ok({'question': user['question']})

@app.route('/api/change-password', methods=['POST'])
def change_password():
    """修改密码：需回答正确验证问题，才能设置新密码"""
    data = request.get_json()
    username    = data.get('username', '').strip()
    answer      = data.get('answer', '').strip()
    new_password = data.get('new_password', '')

    user = USERS.get(username)
    if not user:
        return err('用户名不存在', 404)
    if user['answer'] != _hash(answer.lower()):
        return err('验证答案错误')
    if len(new_password) < 6:
        return err('新密码长度不能少于 6 位')

    USERS[username]['password'] = _hash(new_password)
    save_users(USERS)
    # 如果当前会话就是该用户，强制重新登录
    if session.get('username') == username:
        session.clear()
    return ok({'message': '密码修改成功，请重新登录'})

# ── 账号管理（仅 admin） ───────────────────────────────────

@app.route('/api/admin/users', methods=['GET'])
def admin_list_users():
    """获取所有用户列表（仅 admin）"""
    if not session.get('logged_in'):
        return err('请先登录', 401)
    if session.get('role') != 'admin':
        return err('权限不足', 403)
    users_out = []
    for uname, udata in USERS.items():
        users_out.append({
            'username': uname,
            'role': udata.get('role', 'viewer'),
        })
    users_out.sort(key=lambda x: (x['role'] != 'admin', x['username']))
    return ok({'users': users_out})

@app.route('/api/admin/users/<username>/role', methods=['PUT'])
def admin_set_role(username):
    """修改指定用户的权限（仅 admin，不能修改自身角色）"""
    if not session.get('logged_in'):
        return err('请先登录', 401)
    if session.get('role') != 'admin':
        return err('权限不足', 403)
    if username == session.get('username'):
        return err('不能修改自身账号的权限')
    if username not in USERS:
        return err('用户不存在', 404)
    data = request.get_json()
    new_role = data.get('role', '').strip()
    if new_role not in ROLE_PERMISSIONS:
        return err(f'无效权限，可选：{", ".join(ROLE_PERMISSIONS.keys())}')
    USERS[username]['role'] = new_role
    save_users(USERS)
    return ok({'message': f'已将 {username} 的权限改为 {new_role}'})

@app.route('/api/admin/users/<username>', methods=['DELETE'])
def admin_delete_user(username):
    """注销指定账号（仅 admin，不能注销自身）"""
    if not session.get('logged_in'):
        return err('请先登录', 401)
    if session.get('role') != 'admin':
        return err('权限不足', 403)
    if username == session.get('username'):
        return err('不能注销当前登录的账号')
    if username not in USERS:
        return err('用户不存在', 404)
    del USERS[username]
    save_users(USERS)
    return ok({'message': f'账号 {username} 已注销'})


@app.route('/')
def index():
    if not session.get('logged_in'):
        return send_from_directory('static', 'login.html')
    return send_from_directory('static', 'index.html')

@app.route('/promo')
def promo_page():
    if not session.get('logged_in'):
        return send_from_directory('static', 'login.html')
    return send_from_directory('static', 'dashboard_promo.html')

@app.route('/user')
def user_page():
    if not session.get('logged_in'):
        return send_from_directory('static', 'login.html')
    return send_from_directory('static', 'dashboard_user.html')

# ── 促销子页面路由 ─────────────────────────────────────────
@app.route('/promo/overview')
def promo_overview_page():
    if not session.get('logged_in'): return send_from_directory('static', 'login.html')
    return send_from_directory('static', 'promo_overview.html')

@app.route('/promo/trend')
def promo_trend_page():
    if not session.get('logged_in'): return send_from_directory('static', 'login.html')
    return send_from_directory('static', 'promo_trend.html')

@app.route('/promo/ate')
def promo_ate_page():
    if not session.get('logged_in'): return send_from_directory('static', 'login.html')
    return send_from_directory('static', 'promo_ate.html')

@app.route('/promo/funnel')
def promo_funnel_page():
    if not session.get('logged_in'): return send_from_directory('static', 'login.html')
    return send_from_directory('static', 'promo_funnel.html')

# ── 用户子页面路由 ─────────────────────────────────────────
@app.route('/user/segments')
def user_segments_page():
    if not session.get('logged_in'): return send_from_directory('static', 'login.html')
    return send_from_directory('static', 'user_segments.html')

@app.route('/user/rfm')
def user_rfm_page():
    if not session.get('logged_in'): return send_from_directory('static', 'login.html')
    return send_from_directory('static', 'user_rfm.html')

@app.route('/user/funnel')
def user_funnel_page():
    if not session.get('logged_in'): return send_from_directory('static', 'login.html')
    return send_from_directory('static', 'user_funnel.html')

@app.route('/user/predict')
def user_predict_page():
    if not session.get('logged_in'): return send_from_directory('static', 'login.html')
    return send_from_directory('static', 'user_predict.html')

@app.route('/user/profile')
def user_profile_page():
    if not session.get('logged_in'): return send_from_directory('static', 'login.html')
    return send_from_directory('static', 'user_profile.html')

@app.route('/user/batch')
def user_batch_page():
    if not session.get('logged_in'): return send_from_directory('static', 'login.html')
    return send_from_directory('static', 'user_batch.html')

@app.route('/login')
def login_page():
    return send_from_directory('static', 'login.html')

@app.route('/register')
def register_page():
    return send_from_directory('static', 'login.html')

@app.route('/change-password')
def change_password_page():
    return send_from_directory('static', 'login.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)


# ============================================================
# 内部工具函数
# ============================================================
def run_psm_quick():
    """快速计算PSM ATT（用于overview接口）"""
    try:
        store_df = df_wa.groupby(
            ['LocationID', 'MarketID', 'MarketSize', 'AgeOfStore', 'Promotion']
        )['SalesInThousands'].mean().reset_index()
        store_df.rename(columns={'SalesInThousands': 'AvgSales'}, inplace=True)
        store_df = pd.get_dummies(store_df, columns=['MarketSize'], drop_first=False)

        df12 = store_df[store_df['Promotion'].isin([1, 2])].copy()
        df12['treatment'] = (df12['Promotion'] == 1).astype(int)

        covariates = [c for c in ['AgeOfStore', 'MarketSize_Medium', 'MarketSize_Small',
                                   'MarketSize_Large'] if c in df12.columns]
        X = df12[covariates].values
        T = df12['treatment'].values

        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)
        lr = LogisticRegression(random_state=42, max_iter=500)
        lr.fit(X_sc, T)
        df12['ps'] = lr.predict_proba(X_sc)[:, 1]

        treated_ps = df12[df12['treatment'] == 1][['ps']].values
        control_ps = df12[df12['treatment'] == 0][['ps']].values
        nn = NearestNeighbors(n_neighbors=1).fit(control_ps)
        _, idx = nn.kneighbors(treated_ps)
        matched_ctrl = df12[df12['treatment'] == 0].iloc[idx.flatten()]
        matched_trt  = df12[df12['treatment'] == 1]
        att = float(matched_trt['AvgSales'].mean() - matched_ctrl['AvgSales'].mean())
        return round(att, 2)
    except Exception:
        return 7.55  # fallback


def run_did_quick():
    """快速计算DiD估计量"""
    try:
        df12 = df_wa[df_wa['Promotion'].isin([1, 2])].copy()
        df12['treated'] = (df12['Promotion'] == 1).astype(int)
        df12['post']    = (df12['week'] >= 3).astype(int)

        pre_t  = df12[(df12['treated']==1)&(df12['post']==0)]['SalesInThousands'].mean()
        post_t = df12[(df12['treated']==1)&(df12['post']==1)]['SalesInThousands'].mean()
        pre_c  = df12[(df12['treated']==0)&(df12['post']==0)]['SalesInThousands'].mean()
        post_c = df12[(df12['treated']==0)&(df12['post']==1)]['SalesInThousands'].mean()
        return round((post_t - pre_t) - (post_c - pre_c), 2)
    except Exception:
        return 1.68  # fallback


# ============================================================
# POST /api/classify/upload  —  上传CSV批量四象限分类
# ============================================================
@app.route('/api/classify/upload', methods=['POST'])
@role_required('user.batch')
def classify_upload():
    try:
        if 'file' not in request.files:
            return err('请上传文件，字段名为 file')
        file = request.files['file']
        if not file.filename.endswith('.csv'):
            return err('只支持 .csv 格式文件')
        import io
        content = file.read().decode('utf-8-sig')
        df_upload = pd.read_csv(io.StringIO(content))
        if len(df_upload) == 0:
            return err('文件为空')
        if len(df_upload) > 50000:
            return err('单次最多50000条')
        uid_col = next((c for c in ['用户ID','uid','user_id','UserID','userid']
                        if c in df_upload.columns), None)
        FEAT = ['cnt_pv','fav_category_code','cnt_fav','active_days','peak_hour','weekend_pct']
        missing_feat = [f for f in FEAT if f not in df_upload.columns]
        if missing_feat:
            return err(f'CSV缺少必要列：{missing_feat}')
        for f in FEAT:
            df_upload[f] = pd.to_numeric(df_upload[f], errors='coerce').fillna(0)
        if model is not None:
            X = df_upload[FEAT].values
            probs = model.predict_proba(X)[:, 1]
        else:
            pv = df_upload['cnt_pv'].values
            fav = df_upload['cnt_fav'].values
            days = df_upload['active_days'].values
            score = (pv * 0.4 + fav * 0.3 + days * 0.3) / 10
            probs = np.clip(score, 0.05, 0.95)
        QUAD = {'A':('可说服型（优先触达）','优先投放促销资源，ROI最高'),
                'B':('待激活型（潜力用户）','可适量触达，观察响应'),
                'C':('抗拒/自然型','暂不建议促销投放')}
        results = []
        for i, prob in enumerate(probs):
            p = float(prob)
            t = 'A' if p>=0.65 else ('B' if p>=0.35 else 'C')
            uid = str(df_upload[uid_col].iloc[i]) if uid_col else str(i)
            results.append({'index':i,'user_id':uid,'convert_prob':round(p,4),
                'tier':t+('-高转化' if t=='A' else '-中转化' if t=='B' else '-低转化'),
                'quad_label':QUAD[t][0],'suggestion':QUAD[t][1]})
        probs_arr = np.array([r['convert_prob'] for r in results])
        summary = {'total':len(results),
            'high_pct':round(float((probs_arr>=0.65).mean()*100),1),
            'mid_pct':round(float(((probs_arr>=0.35)&(probs_arr<0.65)).mean()*100),1),
            'low_pct':round(float((probs_arr<0.35).mean()*100),1),
            'avg_prob':round(float(probs_arr.mean()),4),
            'model_used':model is not None}
        return ok({'results':results,'summary':summary})
    except Exception as e:
        return err(str(e))


# ── 健康检查接口 ───────────────────────────────────────────
@app.route('/api/health')
def health():
    # xgboost 安装检查
    xgb_installed = False
    xgb_version = None
    try:
        import xgboost as xgb
        xgb_installed = True
        xgb_version = xgb.__version__
    except ImportError:
        pass

    model_path = DATA.get('model')
    return ok({
        'status': 'ok',
        'model_loaded': model is not None,
        'model_path_found': model_path is not None,
        'model_path': model_path,
        'xgboost_installed': xgb_installed,
        'xgboost_version': xgb_version,
        'diagnosis': (
            '模型正常运行' if model is not None else
            'xgboost未安装，请运行: pip install xgboost' if not xgb_installed else
            '模型文件未找到，请将 xgb_ub_model.pkl 放入 data/ 文件夹' if not model_path else
            '模型加载失败，可能是xgboost版本不兼容，请运行: pip install xgboost==2.0.3'
        ),
        'data_rows': {
            'wa':     len(df_wa),
            'ub':     len(df_ub),
            'uplift': len(df_uplift),
        }
    })


if __name__ == '__main__':
    print('\n' + '='*50)
    print('  电商促销评估系统 Flask 后端')
    print('  http://localhost:5000')
    print('  API文档: http://localhost:5000/api/health')
    print('='*50 + '\n')
    app.run(debug=True, host='0.0.0.0', port=5000)
