# E-commerce Promotion Evaluation System

一个以 Flask、Pandas、scikit-learn 和 XGBoost 构建的电商促销评估与用户行为分析毕业设计。仓库包含 **离线数据处理/建模脚本、可运行的 Flask API、原生 HTML/CSS/JavaScript 页面**，把促销方案对比、用户行为画像和预测演示接到同一个界面。

## 我具体实现了什么

- **促销效果链路：** 对 WA 门店周销售数据做分组统计，并分别实现倾向得分匹配（PSM）与活动前后双重差分（DiD），在页面展示方案均值、周趋势和效果对照。
- **用户行为链路：** 将行为日志聚合为用户级 RFM、浏览/收藏、活跃时间与品类特征；训练 XGBoost 转化模型，产出用户评分与特征重要性，供画像、漏斗和预测页面读取。
- **增量分析链路：** 将促销曝光日志聚合为用户级处理组/对照组，用 S-Learner/T-Learner 估计响应差异并导出 Uplift 分群结果。
- **应用层：** Flask 在启动时加载派生 CSV 和模型，提供 JSON 接口；会话登录与角色权限保护分析/预测接口，浏览器端通过 ECharts 呈现分析结果。

```text
原始数据（不在仓库）
  -> scripts/ 离线清洗、特征工程、统计与模型训练
  -> backend/data/ 派生演示 CSV + 序列化模型
  -> backend/app.py 启动时加载、计算/查询、返回 JSON
  -> backend/static/ 原生页面与 ECharts 图表
```

## 从代码看项目

| 处理环节 | 具体代码 | 输入、输出与实现要点 |
| --- | --- | --- |
| 门店促销统计 | [`wa_analysis.py`](scripts/01_WA分析/wa_analysis.py) | 按促销方案聚合门店销售，做均值、ANOVA/回归与图表。 |
| PSM | [`wa_psm.py`](scripts/02_PSM因果推断/wa_psm.py) | 对方案 1/2 的门店用规模、店龄拟合倾向得分；限制共同支撑域后做最近邻匹配和协变量平衡检验。 |
| DiD | [`wa_did.py`](scripts/01_WA分析/wa_did.py) | 将 Week 1–2 / 3–4 作为前/后期，比较方案 1/2 的差中差，并输出回归及图表。 |
| 用户特征 | [`ub_feature_engineering.py`](scripts/04_UserBehavior分析/ub_feature_engineering.py) | 行为日志按用户聚合成 RFM、浏览/收藏、活跃天数与时间/品类特征，写出 `user_features.csv`。 |
| 转化建模 | [`ub_xgboost_model.py`](scripts/04_UserBehavior分析/ub_xgboost_model.py) | 分层训练/测试、XGBoost、AUC/KS/Lift/特征重要性，导出评分 CSV 与 `joblib` 模型。 |
| 曝光增量 | [`log_uplift_model.py`](scripts/05_日志数据分析/log_uplift_model.py) | 从日志聚合曝光与转化，分别训练 S-/T-Learner，输出每用户 `uplift_t` 和分群 CSV。 |
| Flask 服务 | [`backend/app.py`](backend/app.py) | 启动时预加载数据与模型；API 计算 KPI/趋势/PSM/DiD、查询 RFM/分群、预测及批量上传。 |
| 页面 | [`backend/static/`](backend/static/) | 原生页面以 HTTP/JSON 调 API，并用 ECharts 画趋势、漏斗、ROC 与用户分布。 |

### 后端如何串起这些结果

[`backend/app.py`](backend/app.py) 从 `backend/data/` 寻找并预加载 WA、用户特征、用户评分、特征重要性、Uplift CSV 和模型。促销页读取 [`/api/promo/overview`](backend/app.py#L208)、[`/api/promo/trend`](backend/app.py#L244)、[`/api/promo/ate`](backend/app.py#L277)；用户页读取 [`/api/user/segments`](backend/app.py#L330)、[`/api/user/rfm`](backend/app.py#L384)、[`/api/user/funnel`](backend/app.py#L442) 和 [`/api/user/profile/<id>`](backend/app.py#L661)。单用户 JSON 预测走 [`/api/predict`](backend/app.py#L568)，批量 JSON 最多 10,000 条走 [`/api/predict/batch`](backend/app.py#L611)，CSV 上传分类最多 50,000 行走 [`/api/classify/upload`](backend/app.py#L988)。接口返回统一的 `code` / `data` JSON 包装，异常路径返回 `message`。

登录后 `role_required()` 根据 `admin`、`analyst`、`viewer` 三种角色检查权限点；管理员可管理用户，分析师可用分析、预测和上传接口，观察者只可看促销总览与趋势。演示账号以本地 JSON 持久化，适合本地作品演示，不是生产级身份系统。更细的请求示例见 [后端接口说明](backend/README.md)，项目背景与方法见 [项目笔记](docs-project-notes.md)。

### 代码口径与限制（便于复核）

- WA 门店销售的 PSM/DiD 与曝光日志的 Uplift 来自**不同数据链路**，页面并列展示的是方法演示，不是对同一批用户、同一实验的三次独立验证；因果解释依赖数据质量和识别假设。
- [`/api/user/segments`](backend/app.py#L330) 读取离线 `log_uplift_result.csv` 的 `user_type` / `uplift_t`；[`/api/classify/upload`](backend/app.py#L988) 则按转化评分阈值给上传 CSV 打演示性标签，**没有为这些新用户重新估计 CATE**，不应把上传结果称作因果四象限。
- [`ub_xgboost_model.py`](scripts/04_UserBehavior分析/ub_xgboost_model.py) 的训练列表为 19 个特征，包含购买派生量；Web 预测接口却传 6 个行为特征。公开脚本与在线输入口径需要统一后，才能从原始数据稳定重训并复现在线推理；不能声称该训练脚本完全排除了购买行为或没有标签泄漏风险。
- 当前 `smooth_prob()` 虽接收模型原始概率，但返回分数由手写行为规则计算，未使用该原始概率；在线展示值是**演示评分**，不能当已校准的 XGBoost 概率。部分 API 的 p 值/备用估计值也是示例常量，生产决策前须重新计算与验证。

## 快速开始

```bat
cd backend
python -m venv .venv
.venv\\Scripts\\activate.bat
pip install -r requirements.txt
set FLASK_SECRET_KEY=replace-with-a-long-random-value
python app.py
```

以上是 Windows `cmd.exe` 示例；PowerShell 使用 `$env:FLASK_SECRET_KEY = '...'`，macOS/Linux 使用 `export FLASK_SECRET_KEY=...` 并激活 `.venv/bin/activate`。建议使用与 [固定依赖版本](backend/requirements.txt)兼容的 Python 环境。打开 `http://localhost:5000`。首次本地运行会创建一个未提交的 `backend/users.json`；演示管理员账号为 `demo_admin`，密码为 `demo123`。仅用于本地演示，不要把默认密钥/账号暴露为公网服务。

## 仓库内容与数据说明

- `backend/`：Flask 服务、原生 HTML/CSS/JavaScript 界面、模型与应用所需的派生演示数据。
- `scripts/`：WA Marketing Campaign、UserBehavior、PSM、DiD、Criteo 和日志 Uplift 的离线脚本；[`criteo_conversion_model.py`](scripts/03_Criteo转化预测/criteo_conversion_model.py) 是单独的转化实验，不是 Web 服务当前加载的模型。
- `docs-project-notes.md`：完整项目设计、指标与方法说明。

为避免重复分发原始大数据集和日志，本仓库不含原始数据文件、压缩备份、训练过程的重复输出或个人账号数据库。需要重新训练时，请从相应公开数据来源取得数据，并通过脚本参数或环境变量指定输入路径；不要提交本地下载的数据集。

## 开源协议

本项目代码采用 [MIT License](LICENSE)。数据集、预训练模型及其再分发仍应遵守各自的来源和许可条款。
