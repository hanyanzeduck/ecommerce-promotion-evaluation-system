# E-commerce Promotion Evaluation System

一个以 Flask、Pandas、scikit-learn 和 XGBoost 构建的电商促销评估与用户行为分析系统。系统把转化预测、用户 RFM 分群、PSM、DiD 与 Uplift 分析集中在一个可交互的 Web 界面中。

## 功能

- 单用户与批量转化率预测
- RFM 用户画像、漏斗与四象限 Uplift 分群
- 促销 KPI、销售趋势和 PSM / DiD / Uplift 因果效应对比
- 基于角色的登录与权限示例

## 快速开始

```bash
cd backend
python -m venv .venv
.venv\\Scripts\\activate       # macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
set FLASK_SECRET_KEY=replace-with-a-long-random-value  # macOS / Linux: export FLASK_SECRET_KEY=...
python app.py
```

打开 `http://localhost:5000`。首次本地运行会创建一个未提交的 `backend/users.json`；演示管理员账号为 `demo_admin`，密码为 `demo123`。仅用于本地演示，部署前请修改密钥与账号存储方式。

## 仓库内容与数据说明

- `backend/`：可运行的 Flask 服务、原生 HTML/CSS/JavaScript 界面、模型与应用所需的派生演示数据。
- `scripts/`：WA Marketing Campaign、UserBehavior、PSM、DiD、Criteo 和日志 Uplift 分析脚本。
- `docs-project-notes.md`：完整项目设计、指标与方法说明。

为避免重复分发原始大数据集和日志，本仓库不含原始数据文件、压缩备份、训练过程的重复输出或个人账号数据库。需要重新训练时，请从相应公开数据来源取得数据，并通过脚本参数或环境变量指定输入路径；不要提交本地下载的数据集。

## 开源协议

本项目代码采用 [MIT License](LICENSE)。数据集、预训练模型及其再分发仍应遵守各自的来源和许可条款。
