# 电商促销评估系统 · Flask 后端

## 目录结构

```
backend/
├── app.py                  ← Flask 主程序
├── requirements.txt        ← 依赖包列表
├── README.md               ← 本文件
├── data/                   ← 数据文件目录（需手动放入）
│   ├── WA_Marketing-Campaign.csv
│   ├── user_features.csv          （来自 ub_feature_engineering.py 输出）
│   ├── user_convert_scores.csv    （来自 ub_xgboost_model.py 输出）
│   ├── feature_importance.csv
│   ├── log_uplift_result.csv
│   └── xgb_ub_model.pkl           （来自 ub_xgboost_model.py 输出）
└── static/                 ← 前端文件
    ├── dashboard_promo.html
    └── dashboard_user.html
```

## 快速启动

### 第一步：安装依赖

```bash
pip install flask flask-cors pandas numpy scikit-learn xgboost joblib scipy
```

### 第二步：放入数据文件

把以下文件复制到 `data/` 文件夹：
- `WA_Marketing-Campaign.csv`
- `user_features.csv`（ub_feature_engineering.py 输出）
- `user_convert_scores.csv`（ub_xgboost_model.py 输出）
- `feature_importance.csv`
- `log_uplift_result.csv`
- `xgb_ub_model.pkl`（ub_xgboost_model.py 输出）

### 第三步：启动服务

```bash
cd backend
python app.py
```

启动成功后访问：
- 促销总览页：http://localhost:5000
- 用户分析页：http://localhost:5000/user

---

## API 接口文档

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/health` | GET | 健康检查，查看后端和模型状态 |
| `/api/promo/overview` | GET | 促销总览 KPI（转化率/GMV/ATT/CATE） |
| `/api/promo/trend` | GET | 周度销售趋势（?promo=all/1/2/3） |
| `/api/promo/ate` | GET | PSM/DiD/Uplift 三种方法效应汇总 |
| `/api/user/segments` | GET | Uplift 四象限用户分群数据 |
| `/api/user/rfm` | GET | RFM 散点图数据（?n=500&segment=all/high/mid/low） |
| `/api/user/funnel` | GET | 转化漏斗各层用户数 |
| `/api/model/metrics` | GET | 模型评估指标（AUC/KS/Lift/ROC曲线） |
| `/api/predict` | POST | 单用户实时转化率预测 |
| `/api/predict/batch` | POST | 批量用户转化率预测（最多10000条） |
| `/api/user/profile/<id>` | GET | 单用户 RFM 画像查询 |

### POST /api/predict 请求示例

```json
{
    "cnt_pv": 5,
    "cnt_fav": 2,
    "fav_category_code": 1,
    "active_days": 3,
    "peak_hour": 20,
    "weekend_pct": 0.4
}
```

### POST /api/predict/batch 请求示例

```json
{
    "users": [
        {"user_id": 1001, "cnt_pv": 5, "cnt_fav": 2, "fav_category_code": 1, "active_days": 3, "peak_hour": 20, "weekend_pct": 0.4},
        {"user_id": 1002, "cnt_pv": 0, "cnt_fav": 0, "fav_category_code": 0, "active_days": 1, "peak_hour": 12, "weekend_pct": 0.2}
    ]
}
```

---

## 品类编码对照表

| 编码 | 品类 |
|------|------|
| 0 | 家居用品 |
| 1 | 家用电器 |
| 2 | 服装鞋帽 |
| 3 | 美妆护肤 |
| 4 | 食品饮料 |
| 5 | 电子产品 |
