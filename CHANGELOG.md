# Changelog

## v0.5.0 — 可复现实验平台 (2026-05-14)

**目标**：从 Streamlit 演示版升级为可脚本化、可批量、可追溯的实验平台。

### 新增
- 成本矩阵三模式：`euclidean_fast` / `road_network_time` / `cached_network_time`（parquet 缓存）
- 步行网络接入：OSM walk graph + Dijkstra 真实步行距离
- 需求点道路节点吸附（全局） + 水体过滤（OSM 水域 polygon）
- 轨道容量敏感性分析：保守 ×0.7 / 基准 ×1.0 / 乐观 ×1.2
- `src/app/` 模块拆分：config / data_loader / outputs / pipeline，主流程脱离 Streamlit
- CLI 单场景入口：`python -m src.app.cli --scenario C --seed 42`
- 批量实验入口：`python -m src.app.batch --scenarios A B C --seeds 42 123`
- `outputs/experiments/{id}/summary.csv` + `manifest.json`
- CI：`.github/workflows/ci.yml`（pytest + compileall）
- `pyproject.toml`：pytest / ruff 配置

### 修改
- `requirements.txt`：新增 `scikit-learn>=1.3`、`pyarrow>=14.0`
- `outputs.py` 统一管理 7 类输出文件

### 测试
- 89 测试全过（单元 66 + 集成 5 + CLI 9 + batch 9）

## v0.4.1 — 结果可信度修复与版本稳定 (2026-05-14)

**目标**：修正评价指标口径、固定实验随机性、统一版本信息，使系统达到可复现实验平台标准。

### 修改
- 修正 `compute_evacuation_metrics()` 统计口径：按人数而非需求点个数统计
- `AllocationResult` 增加 `mode_people` / `unassigned_people` 字段
- 方式映射修正：`rail_share = (walk_rail + bus_rail) / total_demand`
- `hash(vid) % 60` → `vehicle_index % 60`，SUMO 出发时间可复现
- 侧边栏增加 `random_seed` 输入框（默认42），需求随机扰动默认关闭

### 文档
- VERSION、streamlit_app.py、场景 YAML、技术蓝图统一为 0.4.1
- 新增 `test_metrics.py` / `test_route_builder_determinism.py`
- 新增 `实施计划-四阶段优化.md`

## v0.4.0 — 轨道协同模型 (2026-05-11)

**目标**：引入轨道交通协同疏散能力，建立 5 方式分配模型和 A/B/C 三方案对比框架。

### 新增
- 轨道站容量模型（静态容量 K_s / 动态处理 Q_s / 线路运能 Q_line）
- 站点压力模型（P_s = 到达/处理，4 级：安全/临界/过载/严重）
- 5 方式协同分配（walk_self / walk_rail / bus_rail / bus_periphery / unserved）
- 负载均衡：最近 3 站选压力最低
- 多轮追踪（RoundResult）与方案对比（A 纯公交 / B 轨道优先 / C 混合协同）
- 利用率水平堆叠条形图、站点压力水平条形图
- 总结报告 8 段模板 + TXT 导出
- 路径审计：路网几何 + 车辆轨迹审计 + CSV 导出
- SUMO 简化路网升级为高精路网（simplify=False），轨迹贴路 9%→54%
- SUMO TraCI 道路封闭
- SUMO 子网裁剪（事件周边 8km）

### 数据
- 轨道站：20 个（1/2/3 号线）
- 轨道线路：3 条
- 公交场站：7 个（Overpass API）

### 已知限制
- 步行接入使用欧氏距离，未使用 OSM walk 网络
- 成本矩阵使用欧氏距离，未使用路网时间矩阵
- 需求点生成未基于道路节点采样
- 需求随机扰动默认开启，实验不可复现

## v0.3.0 — SUMO 动态仿真版本 (2026-05-11)

**目标**：将公交调度方案接入 SUMO，实现真实路网上的动态交通仿真。

### 新增
- OSM → SUMO 路网转换（netconvert 原生导入，191k 路段）
- 调度方案 → SUMO trips（fromLonLat/toLonLat OD 对）
- duarouter 真实路网路由计算
- SUMO headless 仿真执行（tripinfo 输出）
- 车辆轨迹提取（SUMO 内部坐标 → WGS84 转换）
- 多轮循环摆渡（并发发车，支持 max_rounds）
- SUMO 轨迹地图叠加（紫色粗线，hover 显示出发/到达时间）
- 学生利用率水平堆叠条形图（颜色按利用率着色）
- 运行日志折叠面板
- 场景信息摘要（页面初始化展示）
- SUMO 环境配置脚本（setup_sumo_env.sh）

### 修复
- 地图2000+道路合并为单trace，大幅提升渲染性能
- 避难点容量数据修复（按类型赋默认值）
- Python 3.9 类型提示兼容

### 已知限制
- 公交停靠位置为需求点而非公交站
- 行人与公交各自独立计算，未协同分配
- SUMO 道路封闭未通过 TraCI 注入
- 循环轮次上限 3 轮（防止仿真时间爆炸）
- 事件类型已移除"地震"选项

## v0.2.0 — 公交调度优化版本 (2026-05-11)

### 新增
- OR-Tools CVRP 调度求解（需求自动拆分、允许部分服务）
- 公交集结区自动放置、车辆运力管理
- 车辆利用率 + 循环摆渡估算 + 方案对比表
- 公交路线地图叠加（蓝色，与行人黑色路径区分）
- CSV 导出

## v0.1.0 — 最小可运行原型 (2026-05-11)

**目标**：跑通路网加载 → 事件设置 → 需求点 → 最短路径 → 地图展示的完整闭环。

### 新增
- OSM 徐州路网数据下载脚本（56,507 边 / 22,277 节点）
- 道路路网、路网节点的加载和 GeoDataFrame 转换
- 突发事件定义（类型、中心、半径、受影响道路筛选）
- 疏散需求点加载和随机生成
- 避难点加载和最近匹配
- NetworkX 最短路径计算（travel_time 为权重）
- Plotly 交互式地图渲染（路网底图、事件、需求点、避难点、疏散路径）
- Streamlit 侧边栏场景配置 + 一键运行
- 疏散逻辑：危险区外避难点筛选
- 路径计算串行/并行双模式（macOS spawn 开销大，小任务自动串行）
- 数据版本清单（data_manifest_v0.1.json）

### 数据
- 道路路网：OSM 徐州真实数据
- 避难点：OSM 提取 451 个（公园/学校/体育馆等），按类型估算容量
- 需求点：30 个，围绕事件中心生成（10,420 人）
- 地铁站：手动录入 20 个（1/2/3 号线）

### 已知限制
- 单事件点，不支持多事件
- 无公交车辆调度（待 v0.2.0）
- 无动态仿真（待 v0.3.0）
- 无轨道交通协同（待 v0.4.0）
