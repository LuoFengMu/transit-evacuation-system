# 城市大客流公交-轨道协同疏散仿真系统

面向大型活动散场大客流场景的公交-轨道-步行多方式协同疏散仿真系统。以徐州市为示范城市，基于真实路网和交通数据，利用运筹优化和交通仿真技术，评估不同疏散方案的效果。

**核心逻辑**：轨道交通承担主干大容量疏散，公交负责接驳补盲和压力分流，步行负责短距离接入，系统通过协同分配和仿真评价选择最优疏散方案。

## 当前版本

**v0.5.0** — 可复现实验平台

## 技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| 路网 | OSMnx + NetworkX | 道路网络加载、最短路径 |
| 公交站 | Overpass API | 徐州真实公交站点 |
| 轨道 | 手动录入 + 公开资料推算 | 20站(1/2/3号线) + 容量模型 |
| 调度 | OR-Tools 9.x | CVRP 公交车辆路径优化 |
| 仿真 | SUMO 1.26 + TraCI | 动态交通仿真 + 道路封闭 |
| 可视化 | Streamlit + Plotly | Web交互界面 + 地图 |
| 地理 | GeoPandas + Shapely + pyproj | 空间数据处理 |

## 环境安装

### 依赖

```bash
pip install -r requirements.txt
```

主要依赖：`osmnx` `networkx` `ortools` `geopandas` `streamlit` `plotly` `pyproj` `sumolib`

### SUMO 安装

macOS:
```bash
brew install sumo
# 或者安装到用户目录
```

确认 SUMO 可用：
```bash
sumo --version
```

### 测试

```bash
python -m pytest tests/ -v
```

## 数据准备

系统已包含以下预处理数据：

| 数据 | 文件 | 说明 |
|------|------|------|
| 路网 | `data/osm/xuzhou_road_network.graphml` | OSM 徐州路网 |
| 需求点 | `data/processed/demand_points_v0.1.geojson` | 30个需求点 |
| 公交站 | `data/processed/bus_stops_v0.3.geojson` | 95个公交站 |
| 轨道站 | `data/processed/rail_stations_v0.4.geojson` | 20个轨道站 |
| 轨道线路 | `data/processed/rail_lines_v0.4.csv` | 3条线路 |
| SUMO路网 | `sumo/networks/xuzhou_full_v2.net.xml` | 高精路网(54k边) |

## 启动

```bash
streamlit run streamlit_app.py
```

默认侧边栏参数：
- 事件地点：彭城广场
- 疏散人数量级：3万人
- 随机种子：42（固定可复现）
- 需求扰动：关闭
- 公交调度/轨道协同/SUMO仿真：启用

选择场景配置后侧边栏参数自动填充默认值。

## 系统架构

```
数据层 → 场景层 → 接入层(步行) → 协同层(5方式分配) →
调度层(OR-Tools CVRP) → 仿真层(SUMO+TraCI) → 轨道层(容量/压力) →
评价层(指标/对比) → 可视化层(Streamlit)
```

## 模块结构

```
src/
├── network/     # 路网加载与最短路径
├── demand/      # 需求点生成与缩放
├── walking/     # 步行接入矩阵
├── dispatch/    # OR-Tools 公交调度
├── rail/        # 轨道容量与协同分配
├── simulation/  # SUMO 仿真
├── evaluation/  # 评价指标、报告、审计
└── visualization/ # 地图与图表
```

## 方案对比

| 方案 | 模式 | 说明 |
|------|------|------|
| A | 纯公交 | 需求点→公交→避难点 |
| B | 轨道优先 | 近距离步行进站，其余公交接驳 |
| C | 混合协同 | 5方式协同分配，站点过载分流 |

## 输出

每次运行自动保存至 `outputs/runs/{run_id}/`：

| 文件 | 内容 | 说明 |
|------|------|------|
| `config.yaml` | 运行配置快照 | 始终生成 |
| `scenario.yaml` | 场景配置副本 | 始终生成 |
| `run_meta.json` | 运行元信息 | 始终生成 |
| `metrics.json` | 方案对比指标 | 轨道协同时包含数据，否则为 `{}` |
| `report.txt` | 文本摘要报告 | 始终生成 |
| `dispatch_summary.csv` | 调度结果 | 公交调度成功时有数据，否则为空表 |
| `station_pressure.csv` | 站点压力 | 轨道协同时有数据，否则为空表 |

## 已知限制

- 默认仍优先使用快速模式，路网时间矩阵和步行网络模式运行成本较高
- 无背景交通流模拟
- SUMO 出发时间为车辆序号取模，非真实时刻表
- 轨道站容量基于公开信息推算，非实测数据
- 已支持单场景 CLI 与批量实验运行；实验结果写入 `summary.csv` 与 `manifest.json`

## 命令行实验

单场景实验可脱离 Streamlit 运行：

```bash
.venv/bin/python -m src.app.cli --scenario C_hybrid_cooperative.yaml --seed 42 --no-sumo
```

常用参数：

```text
--scenario          场景 YAML 文件名或完整路径
--seed              随机种子
--demand            疏散人数量级
--cost-mode         euclidean_fast / road_network_time / cached_network_time
--no-bus            禁用公交调度
--no-rail           禁用轨道协同
--no-sumo           禁用 SUMO 仿真
--capacity-factor   0.7 / 1.0 / 1.2
--output            自定义输出目录
```

## 版本路线

| 版本 | 主题 | 状态 |
|------|------|------|
| v0.4.1 | 可信度修复与稳定 | ✅ |
| v0.5.0 | 可复现实验平台：CLI + 批量实验 + 实验清单 | ✅ 就绪 |
| v0.6.0+ | 真实数据接入与工程服务化（按需） | 规划中 |
| v1.0.0 | 论文/答辩稳定版 | 规划中 |

## 文档

- [技术蓝图与开发路线](版本内容管理/技术蓝图与开发路线.md)
- [项目代码优化与修改建议](版本内容管理/项目代码优化与修改建议.md)
- [实施计划-四阶段优化](版本内容管理/实施计划-四阶段优化.md)
