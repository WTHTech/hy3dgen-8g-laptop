# DFM 可制造性检查 — 方案与落地指南

> 本文档面向 Phase 2 开发，整理 DFM 全链路：工具选型 → 检查项 → 修复策略 → 评分闭环。
> 原则：能用现成库的不自己写算法，创新点集中在**闭环编排、评分体系、工艺规则配置**。

> **文档口径（2026-08-05）**：下文同时包含项目目标与当前实现。只有”当前可靠”项目可以输出 PASS/FAIL 并参与评分；未完成或算法尚未验证的项目必须输出 `UNKNOWN`，工艺不适用项输出 `NOT_APPLICABLE`。不得把跳过、依赖异常或占位实现记为通过。

## 当前实现基线

| 层级 | 当前可靠 | 当前降级为 `UNKNOWN` / `N/A` |
|------|----------|-------------------------------|
| 快速层 | P0、P1、P2、P3、G1~G6、S1（11 项） | G2 面数超过同步上限或依赖异常时为 `UNKNOWN` |
| 精检层 | W1 射线抽样估算、D2 多部件最近距离 | W2~W4、C1~C2、S2~S5、D1、D3 |
| 修复层 | 轻量修复(5步 pymeshlab)、中度修复(补洞+非流形+MeshFix CLI)、P1/P3底座自动生成(`pedestal.py`)、G6碎面自动清理(`fragment_cleaner.py`)、候选网格事务验收/回滚(`repair_policy.py`)、独立进程硬超时与可恢复状态(`repair_worker.py`)、复检闭环、多轮重试编排（底座→碎片→拓扑修复→追加清理→底座重试，每类≤2次）、FALLBACK 兜底 | 重度修复(V-HACD 重网格化) |
| 输出 | 四态结果、配置化加权评分、JSON、`prepared_mesh`、`RepairResult`、`PedestalResult`、`FragmentCleanResult`、CuraEngine FDM 仿真 `SliceResult`、raw mesh 可恢复批量验证(`batch_dfm_benchmark.py`) | 缺陷热力图、目标打印机可直接上机配置 |
| AI预处理 | `image_preprocess/`：OpenAI gpt-image-2 + 火山方舟 Seedream 5.0 Pro 底座图像编辑 | 正式 API 密钥验证、批量预处理 |

`DFMReport.prepared_mesh` 是统一到打印坐标、缩放至毫米并放置到平台后的实际检查对象，后续导出和精检必须使用它，不能再使用原始归一化网格。Hunyuan 归一化输出按配置 `platform.normalized_up_axis: y` 先从 Y-up 旋转为打印坐标 Z-up，再按旋转后的 Z 高度缩放至 `target_height` 并落到 Z=0；已有 STL/毫米模型传 `input_units='mm'`，保持原始朝向且不会再次缩放。`auto` 遇到小尺寸歧义会返回 `INCOMPLETE`，不会擅自放大。这里完成的是**已知生成器坐标约定转换**，不等于按支撑、悬垂和表面质量搜索最佳打印朝向。`PASS` 表示所有适用项目都有结论且通过；存在 `UNKNOWN` 时报告状态为 `INCOMPLETE`。`Hunyuan3DGenerator.export_stl/obj/glb(report, ...)` 默认只接受 `PASS` 报告，诊断导出必须显式传 `allow_failed=True`。

切片仿真必须从报告对象进入，不能再次传原始生成网格：

```python
from dfm import CuraSlicer, DFMChecker, DFMRules

rules = DFMRules()
report = DFMChecker(rules).check_quick(mesh, input_units='normalized')
slice_result = CuraSlicer(rules=rules).slice_prepared(
    report, 'output/model.gcode'
)
```

默认只允许 `PASS` 报告进入切片。通用 profile 生成的 G-code 标记为
`simulation_only=True`，只用于可切片性、层数、时间和耗材估算，不能直接发送打印机。

---

## 一、工具矩阵

| 工具 | 定位 | 核心能力 | 安装方式 |
|------|------|---------|---------|
| **trimesh** | 主力分析库 | 水密/连通分量/截面/射线/包围盒 | `pip install trimesh`（已装） |
| **pymeshlab** | 高级检查+修复 | 自相交选择、非流形选择、法向统一、几何度量 | `pip install pymeshlab`（已装） |
| **MeshFix CLI** | 批量修复兜底 | 非流形/破洞/重叠面一键修复，无 GUI | 下载二进制，subprocess 调用 |
| **CuraEngine** | FDM 切片仿真 | 验证模型能否正常切片，输出层数/耗时/耗材/刀路边界 | Windows 5.13 运行时已接入，`subprocess` 调用；服务器 Linux 构建待部署 |

> **底层依赖**：Shapely（平面几何操作）、scipy（连通域/空间计算）、numpy（向量计算）。当前版本的 trimesh 没有 `is_self_intersecting`，当前版本的 pymeshlab 也没有 `compute_thickness` / `get_non_manifold_edges`；代码不能引用不存在的 API。

---

## 二、检查项全景

### 2.1 前置处理

| # | 检查项 | 工具 | 关键点 |
|---|--------|------|--------|
| P0 | **坐标/单位标准化** | trimesh | `normalized` 按已配置源竖直轴转为 Z-up 后定高；`mm` 保持原朝向；歧义输入不自动猜测 |
| P1 | **底面平台校验** | trimesh | 只累加与 Z=0 共面的真实三角面投影；点/边接触面积为 0，禁止用分散最低点凸包代替 |
| P2 | **外框超限** | trimesh `bounds` | AABB 超过打印机成型尺寸 → 直接拒绝 |
| P3 | **静态稳定性** | scipy ConvexHull | 重心XY投影是否位于底面支撑凸包的安全裕量内；裕量不足有倾倒风险 |

### 2.2 几何拓扑（致命缺陷 → 阻断级）

| # | 检查项 | 工具 | 方法 |
|---|--------|------|------|
| G1 | **水密性** | trimesh `is_watertight` | 一行判定；不通过 → 阻断 |
| G2 | **自相交** | pymeshlab `compute_selection_by_self_intersections_per_face()` | 统计被选中的自相交面；不通过 → 阻断；超大网格转异步任务 |
| G3 | **非流形边** | trimesh `edges_unique_inverse` | 统计被 ≥3 个面共用的边；不通过 → 阻断 |
| G4 | **翻转法向** | trimesh `is_winding_consistent` + 有符号体积 | 相邻面绕序不一致或水密网格体积为负 → 阻断 |
| G5 | **重复/零面积面片** | numpy + trimesh `area_faces` | 面顶点索引排序去重；按模型尺度设置退化面积容差 |
| G6 | **孤立游离碎面/悬空分量** | trimesh `split()` | 小分量标记碎面；任何未接触平台的独立分量直接失败 |

### 2.3 壁厚与结构（警告/阻断，按工艺可配置）

| # | 检查项 | 工具 | 方法 | 参数 |
|---|--------|------|------|------|
| W1 | **最小壁厚（当前为抽样估算）** | trimesh 射线 | 表面确定性采样，按 `index_ray` 将命中点配回发射原点，取每条射线最近有效命中 | FDM ≥ 1.2mm（喷嘴直径×3） |
| W2 | **最小特征尺寸** | pymeshlab + SDF | 骨架分析检测细剑尖/飘带末梢等脆弱结构 | 小于喷嘴直径 → 阻断 |
| W3 | **拉丝/细长比** | trimesh 截面 + Shapely | 柱体高度/最小宽度 > 10~15 → 振动变形风险 | 按材料设定 |
| W4 | **镂空点阵壁厚** | trimesh SDF + 体素化 | 距离场法度量两实体面间距，精准定位薄弱点 | 体素精度 0.1mm |

### 2.4 空腔检测

| # | 检查项 | 工具 | 方法 |
|---|--------|------|------|
| C1 | **封闭空腔** | 体素化 + scipy 洪泛 | 对模型包围盒外扩后的**空域**从边界做 3D 洪泛；未连通到外界的空域才是封闭空腔。当前未实现，返回 `UNKNOWN` |
| C2 | **排液孔**（光固化） | 同上 + 规则 | 中空模型必须有排液孔，否则"吸力杯"效应 → 打印失败 |

> 原“筛选实体内部点再做连通域”的方案检测的是实体材料，不是空腔，已停用。后续必须对外部空气与内部空域的连通性进行判断。

### 2.5 悬垂与支撑

| # | 检查项 | 工具 | 方法 |
|---|--------|------|------|
| S1 | **悬垂角** | numpy 法向量 + 面积 | `normal_z < -cos(临界角)` 的朝下面为悬垂；排除 Z=0 底面，并按表面积而非面片数量统计。竖直侧壁不属于悬垂 |
| S2 | **桥接检测** | trimesh 截面分析 | 水平悬空两点间跨度 > 阈值 → 拉丝风险（FDM 独有痛点） |
| S3 | **悬空孤岛** | trimesh 分层切片 + 连通比对 | 某层新出现的轮廓在下方无对应 → 完全悬空，打印坍塌 |
| S4 | **最小支撑接触面积** | numpy + trimesh 面片遍历 | 悬垂面投影面积累加 < 阈值 → 支撑脱落 |
| S5 | **支撑去除干涉** | pymeshlab 布尔交集 | 支撑网格 ∩ 模型 → 嵌入风险，拆支撑刮伤零件 |

### 2.6 尺寸与装配

| # | 检查项 | 工具 | 方法 |
|---|--------|------|------|
| D1 | **内孔最小孔径** | trimesh 截面 + Shapely `interiors` | 分层切→提取内轮廓→计算最小宽度 < 喷嘴直径 → 堵死 |
| D2 | **装配间隙** | trimesh `ProximityQuery` | 多零件间最近距离 < 0.2~0.3mm → 粘连 |
| D3 | **螺纹特征** | trimesh 高频切片 + scipy 周期分析 | 牙距/牙高小于成型极限 → 螺纹失效 |

---

## 三、分层质检流水线

批量处理时按优先级分两级，避免对致命坏模型浪费高精度检测时间：

```
┌─ 第一级：快速粗筛（同步；G2 耗时随面数增长）───────────┐
│  P0 单位标准化 → P1 底面平台 → P2 外框超限 → P3 稳定性  │
│  G1~G6 拓扑 → S1 悬垂角                                  │
│  FAIL/UNKNOWN 阻断 → 不进入精检；超大模型转服务器异步任务 │
└───────────────────────────────────────────────────────┘
                         ↓ 通过
┌─ 第二级：精细精检 (pymeshlab, 秒级) ───────────────────┐
│  W1 壁厚抽样 → D2 装配间隙                               │
│  其余候选项在算法与真值集验收前统一返回 UNKNOWN           │
└───────────────────────────────────────────────────────┘
                         ↓ 通过/修复后
┌─ 第三级：修复 + 仿真 ──────────────────────────────────┐
│  [P1/P3]底座自动生成 → [G6]碎片自动清理 → pymeshlab     │
│  → MeshFix → 复检闭环 → 仅 PASS → slice_prepared()      │
│  → 解析 SliceResult（层数/耗时/耗材/刀路边界）            │
└───────────────────────────────────────────────────────┘
```

---

## 四、分级修复策略

| 级别 | 操作 | 工具 | 示例 | 状态 |
|------|------|------|------|:--:|
| **底座** | P1/P3 失败时自动生成适配底座（disc/block/auto），多级布尔融合后复检 | trimesh + pymeshlab | 仅用于 P1/P3；底座厚度、外扩、锥角可配置 | ✅ |
| **清理** | G6 失败时自动删除孤立/悬浮小碎片，保守策略保护嵌入配件 | trimesh split + ProximityQuery | 仅用于 G6；同时满足面积比、绝对面积、最长边三项限制且无干涉才删 | ✅ |
| **轻量** | 按毫米容差合并重叠顶点、删除重复/零面积面、统一并校正整体法向、清理孤立顶点 | pymeshlab 滤镜 | 仅用于 G4/G5；每轮后复检 | ✅ |
| **中度** | 按边界环边数填补破洞、修复非流形边/顶点 | pymeshlab | 仅用于 G1/G2/G3；复检仍失败后才考虑 MeshFix | ✅ |
| **中度** | 黑盒修复兜底 | MeshFix CLI | 自带子进程超时、自动清理临时文件 | ✅ |
| **重度** | V-HACD 重网格化、网格简化 | trimesh / MeshFix | 超大模型减面至可打印量级 | 🔲 预留 |
| **兜底** | 无法修复 → 退回生成阶段换 seed 重新生成 | 自研调度逻辑 | 对应方案 4.1.1「可打印性约束前置」 | ✅ |

修复闭环 API：`DFMRepairer.repair_and_recheck(mesh, checker, input_units='normalized')` →
初始快速检查 → `UNKNOWN` 转异步/人工复核；P1/P3 进入底座自动生成(`PedestalGenerator`)并复检；G6 进入碎片自动清理(`FragmentCleaner`)并复检；碎片清理暴露 P1/P3 时自动重试底座；G4/G5 进入轻量修复并复检；G1/G2/G3 进入中度修复并复检；拓扑修复后仍有 G6 或 P1/P3 时自动触发追加清理/底座重试（每类最多 2 次）；仍失败时才运行 MeshFix。`RepairResult.success` 只表示快速拓扑层修复成功，仍须运行精检与切片。

每次候选修复复检后都经过 `repair_policy.evaluate_repair_candidate()` 事务门：候选不得增加 FAIL 数量、降低快速层分数或引入新的 UNKNOWN。通过后才替换当前网格；拒绝时记录 `事务性验收回滚` 动作和 `rolled_back` 进度事件，保留阶段前网格。拓扑阶段被回滚后立即停止由该候选派生的中度/碎片/底座操作，避免在不存在的新缺陷上继续修改。

底座、碎片清理及 pymeshlab 轻量/中度修复默认由 `dfm.repair_worker` 在独立子进程中执行，单阶段时限由 `repair.repair_stage_timeout_seconds` 配置；MeshFix 继续使用自身的命令行超时。超时或工作进程结果损坏时，父进程终止该阶段并保留阶段开始前的最后有效网格，不把半成品覆盖为成功。返回的 `RepairResult` 会设置 `timed_out=True`、`timeout_stage` 和 `resume_from_stage`，调用方可用 `progress_callback` 接收 `started`、`completed`、`failed`、`timed_out` 结构化事件并从记录阶段续跑。

当前修复器是纯几何修复器，不保留 UV、材质、纹理或顶点/面颜色。检测到外观属性时会安全拒绝；完整版服务器应在纹理生成前完成几何修复，或另建属性重投影流程。

修复完成后**必须重新跑全部检测**（检查→修复→复检闭环），确认缺陷消除且无新增问题。

---

## 五、可打印性评分体系

每个已执行的维度输出 0~100 分，最后按 `dfm_config.yaml` 加权汇总。`UNKNOWN` 和 `NOT_APPLICABLE` 不计分，缺失维度也不会自动按 100 分补齐。总分必须与报告的完整性状态同时展示，不能把 `INCOMPLETE: 100` 宣传成“全部通过”。

| 维度 | 权重 | 评分逻辑 |
|------|:--:|------|
| 水密/拓扑 | 25% | 水密+无非流形=满分，每项缺陷扣分 |
| 壁厚 | 25% | 全局最小壁厚/目标壁厚 × 100，< 阈值直接 0 |
| 悬垂/支撑 | 20% | 悬垂面占比越少分越高，> 30% 面积 ≤ 60 分 |
| 尺寸合规 | 15% | 外框/内孔/间隙均合规 = 满分 |
| 特征/结构 | 10% | 最小特征尺寸/喷嘴直径 × 100 |
| 空腔 | 5% | 无封闭空腔或有排液孔 = 满分 |

**输出**：总分 + 各维度红黄绿灯仪表盘（答辩展示用）。

---

## 六、输出能力

| 输出类型 | 内容 | 用途 |
|----------|------|------|
| **结构化报告** (JSON，已实现基础版) | PASS/FAIL/UNKNOWN/N/A、量化指标、毫米变换、处理后网格摘要 | 流水线自动化 |
| **缺陷可视化** | 在 STL 上标注破洞(红)、相交面(橙)、薄壁(蓝) | 快速定位 |
| **热力图** (PNG) | 壁厚、悬垂角、特征尺寸云图 | 答辩展示 |
| **切片仿真报告**（已实现基础版） | 成功状态、错误/警告、层数、层高、耗材长度/质量、打印时长、刀路边界、挤出移动数、引擎版本 | 可切片性与估算决策 |

### 6.1 CuraEngine FDM 切片仿真（已实现基础版）

- **本地运行时**：Windows CuraEngine 5.13 位于 `.cache/tools/curaengine/5.13.0/windows-x64/`，不提交 Git；版本、来源和 SHA-256 记录在 `tools/curaengine/runtime-manifest.json`
- **发现顺序**：构造参数显式路径 → `CURA_ENGINE_PATH` / `CURA_ENGINE_RESOURCES` → 项目 `.cache` → 系统 `PATH`
- **命令栈**：加载全局 `fdmprinter` 和单挤出机 `fdmextruder` 定义，并显式传入定义搜索目录；当前只支持 FDM、单挤出机和 STL 输入
- **安全门禁**：业务链路只能调用 `slice_prepared(report, ...)`，默认仅接受 `DFMReport.status == 'PASS'`；`allow_failed=True` 只用于定位失败模型为何仍能或不能被切片
- **毫米语义**：底层 `slice()` 只接受 `input_units='mm'`；归一化 AI 网格必须先经过 DFM，使用 `report.prepared_mesh`
- **指标解析**：优先采用最终 `TIME_ELAPSED`，从完整 E 轴状态计算实际挤出并扣除回抽；同时校验有效打印层、有效挤出路径和刀路边界
- **文件安全**：使用独立临时目录，成功后原子替换目标 G-code；超时、异常或失败都会清理临时文件，不会把旧输出误判为成功
- **用途边界**：当前通用配置统一标记 `simulation_only=True`。它可证明模型能进入 Cura 切片并给出同配置下的相对估算，但没有目标打印机、材料和质量 profile，不能直接上机，也尚未自动并入 DFM 评分

本地集成验收（同一通用仿真 profile）：

| 样本 | 入口与状态 | 层数 | 时间 | 耗材 | 结论 |
|------|------------|----:|-----:|-----:|------|
| 20 mm 立方体 | `slice()`，毫米输入 | 100 | 约 25.08 min | 2001.56 mm / 5.97 g | Cura 运行时、命令栈和指标解析通过 |
| 当前图生 3D 冒烟模型 | DFM 为 `FAIL`（P1/G5），仅以 `allow_failed=True` 诊断 | 500 | 约 1354.52 min | 129562.72 mm / 386.427 g | 能切片不等于可制造；默认业务入口会阻断 |

---

## 七、工程规范

### 阈值配置化

所有阈值抽到配置文件，按工艺一键切换：

```yaml
# dfm_config.yaml（节选，实际配置按分组书写）
process: FDM
overhang:
  critical_angle: 45
  max_overhang_area_ratio: 0.30
  bridge_max_span: 15
repair:
  merge_close_tolerance: 0.05
  max_hole_edges: 30
  meshfix_timeout_seconds: 120
slicer:
  cura_timeout_seconds: 120
  filament_density_g_cm3: 1.24
performance:
  wall_sample_count: 800
  self_intersection_max_faces: 200000
sla_overrides:
  critical_angle: 30
  require_drain_hole: true
```

### 其他规范

- **服务器批处理（目标）**：高分辨率模型进入异步任务，设置面数、体素数、超时和内存护栏；当前本地 mini 调试仍为同步调用
- **本地 raw mesh 批处理（已实现）**：`scripts/batch_dfm_benchmark.py` 支持输入/配置指纹、阶段原子落盘（state.json、report.json、prepared STL）、`--resume` 复用已完成阶段、`--retry-timeouts` 显式重试超时阶段、单样本失败隔离及 CSV/JSON 汇总；当前为顺序执行且不包含图生阶段，24 图 manifest 驱动图生模式仍待扩展。A01~E01 五图重跑结果：3 个 COMPLETED、2 个 COMPLETED_WITH_TIMEOUT、0 个样本级 FAILED
- **Cura 服务器部署（目标）**：固定 Linux x64 CuraEngine 版本、资源目录和 SHA-256，容器内使用只读运行时；为每种目标打印机/材料建立并验收独立 profile
- **异常语义**：损坏输入为 `FAIL`；依赖异常或计算未完成为 `UNKNOWN`；不能 catch 后标为 PASS
- **格式支持**：优先 STL/OBJ，后续扩展 3MF/STEP（需 trimesh + pythonocc）

---

## 八、落地优先级

| 优先级 | 内容 | 预计工时 | 状态 |
|:--:|------|:--:|:--:|
| **P0** | P0~P3、G1~G6、S1、Y-up→Z-up 坐标标准化、四态报告、配置校验、JSON、几何真值回归测试 | — | ✅ 已完成 |
| **P1** | 底座自动生成(`pedestal.py`)、碎片自动清理(`fragment_cleaner.py`)、候选网格事务验收/回滚、修复阶段硬超时/恢复信息、修复闭环多轮重试编排 | — | ✅ 已完成 |
| **P2** | W1 增加薄壳/复杂手办真值集；G2 高面数性能基准；缺陷可视化；AI 底座图像预处理集成验证 | 1~2 天 | 🔄 W1 真值集、G2 性能验证、可视化待做；AI 底座图像预处理模块已实现，真实 API 验证待完成 |
| **P3** | C1 空域洪泛、D1 任意方向孔、S2/S3 层间支撑、局部骨架分析 | 按规则逐项验收 | 🔲 待开始 |
| **P4** | 基于悬垂/支撑/表面质量的自动朝向优化、manifest 驱动图生批处理、并行批处理、光固化模式、服务器 Linux CuraEngine 部署 | 按需 | 🔄 已完成已知坐标系转换、Windows CuraEngine 5.13 基础仿真、raw mesh 可恢复批处理和图像底座预处理模块；朝向搜索、并行/manifest 图生模式、目标打印机 profile、Linux 部署待做 |

---

## 九、AI 底座图像预处理（新增）

在进入图生 3D 前，可选择对输入图片进行 AI 底座编辑，降低 P1（无有效平台接触面）的发生率。

| 项目 | 内容 |
|------|------|
| **模块** | `image_preprocess/` — 分析器、蒙版生成、流水线编排、提供方适配器 |
| **提供方** | OpenAI `gpt-image-2`（官方端点）、火山方舟 Seedream 5.0 Pro（独立端点） |
| **策略** | `auto`：仅手办类且底部风险高时编辑；`always`：强制添加底座；`never`：保持原图 |
| **安全边界** | API Key 只读环境变量，异常/回退不泄密；输出无效或主体消失时自动回退原图；不阻断后续 DFM |
| **验证** | 104 项全量测试通过；真实 API 验证：OpenAI 官方端点密钥不可用、Seedream 首次参数已修复，等待用户授权第二次真实验证 |
| **接入** | `Hunyuan3DGenerator.image_to_prepared_3d(image_preprocessor=...)` 可选接入 |

> AI 底座编辑只是降低 P1 风险的预处理手段，不能代替后续 DFM 检查和修复闭环。

## 十、与 Hunyuan3D 生成的对接

### 快捷入口

```python
# 一行完成：生成 → DFM → 修复 → 导出毫米 STL/GLB
result = gen.image_to_prepared_3d(
    'demo.png', target_height=100,
    export_stl='output.stl', seed=0,
)
# result['report']         → DFMReport（最后复检状态）
# result['prepared_mesh']  → 毫米网格（导出和切片共用）
# result['ready_for_slicing'] → bool（快速层 PASS？）
# result['exported_paths'] → {'stl': 'output.stl', ...}
```

### 完整数据流

```
Hunyuan3D generate_shape()
        │
        ▼
    trimesh.Trimesh  (归一化坐标)
        │
        ▼  image_to_prepared_3d() 快捷入口 ──────────────┐
        │                                                 │
┌───────▼──────────────┐                                  │
│  DFMChecker.check_quick()  │  ← dfm/checker.py          │
│  P0~P2 + G1~G6 + S1       │     input_units='normalized'│
│        │                   │                             │
│        ▼                   │                             │
│  DFMReport {               │                             │
│    status, score,          │                             │
│    prepared_mesh  ← 毫米   │                             │
│  }                         │                             │
│        │                   │                             │
│    ┌─── PASS ──→ export_stl/obj/glb(report) ✅          │
│    │          │  默认阻断 FAIL/INCOMPLETE                │
│    │          └→ CuraSlicer.slice_prepared(report)       │
│    │               → SliceResult                         │
│    │                                                     │
│    ├─── INCOMPLETE ──→ 异步精检/人工复核                │
│    │                                                     │
│    └─── FAIL ──→ DFMRepairer.repair_and_recheck() ✅    │
│                      │                                   │
│                  ┌─── 修复成功 → 复检 PASS → 导出+切片   │
│                  └─── 修复失败 → FALLBACK → 换 seed      │
└──────────────────────────────────────────────────────────┘
