# DFM 批量验证计划与目录规范

> 建立日期：2026-08-04  
> 当前阶段：目录与 manifest 已建立，底座(`pedestal.py`)、碎片清理(`fragment_cleaner.py`)、P3 静态稳定性、Hunyuan Y-up→打印 Z-up 标准化、修复阶段硬超时/恢复状态和多轮重试编排已完成；A01~E01 五图已完成 mini 冒烟生成/初检，并使用可恢复脚本复用 raw mesh 跑完 DFM/修复/复检/切片门禁。24 图正式样本和 manifest 驱动的图生批处理仍待完成。  
> 目标：获得可复现的修复前后原始数据、区分代码问题与模型质量问题、形成比赛可用的量化证据。

## 一、当前验证边界

本轮使用本地 Hunyuan3D-2mini 进行调试，先验证图生 3D → DFM 快速检查 → 修复 → 复检 → CuraEngine 仿真的工程闭环。后续服务器完整版必须复用同一数据集、清单和输出结构，才能进行公平对比。

当前可作为可靠结论的快速检查项（11 项）：

- P0：坐标、单位与输入标准化
- P1：底面平台
- P2：外框超限
- P3：静态稳定性（重心投影安全裕量）
- G1~G6：水密、自相交、非流形边、法向、重复/退化面、孤立组件
- S1：悬垂角

对应的自动修复能力：

- P1/P3 → `PedestalGenerator`：自适应底座生成与实体融合（disc/block/auto，多级布尔回退）
- G6 → `FragmentCleaner`：孤立碎面保守清理（三项面积限制 + AABB 距离保护）
- G4/G5 → pymeshlab 轻量修复
- G1/G2/G3 → pymeshlab 中度修复 → MeshFix CLI 兜底

当前基础精检项为 W1 和 D2；W2~W4、C1~C2、S2~S5、D1、D3 尚未完成真值验收，必须保留为 `UNKNOWN` / `NOT_APPLICABLE`，不能计入通过率，也不能登记成模型质量缺陷。

## 二、目录结构

```text
D:\AI文创3D打印\
├── data/
│   └── dfm_benchmark_v1/
│       ├── manifest.csv                    # 24 张图片的元数据和预期风险清单
│       ├── images/                         # 原始输入图，本地保存且不提交 Git
│       │   ├── A_baseline/                 # 正向基准 4 张
│       │   ├── B_platform_contact/         # 平台接触与重心 4 张
│       │   ├── C_thin_features/            # 细小结构 4 张
│       │   ├── D_overhang_components/      # 悬垂和多组件 4 张
│       │   ├── E_cavity_assembly/          # 孔洞、空腔和装配 4 张
│       │   └── F_input_controls/           # 同主体输入对照 4 张
│       └── derived/                        # 脚本生成的尺寸/色彩模式等派生输入
├── scripts/
│   └── batch_dfm_benchmark.py              # 已实现：raw mesh 可恢复 DFM 批量验证入口
├── benchmarks/
│   └── dfm_v1/
│       ├── runs/                           # 每张图、每个 seed 的大体积逐次产物
│       │   └── <image_id>/<run_id>/
│       │       ├── input_meta.json
│       │       ├── raw_mesh.glb
│       │       ├── initial_prepared.stl
│       │       ├── initial_report.json
│       │       ├── repair_result.json
│       │       ├── final_prepared.stl
│       │       ├── final_report.json
│       │       ├── slice_result.json
│       │       ├── run.json
│       │       └── run.log
│       ├── summaries/                      # 可提交的 CSV/JSON 汇总
│       │   ├── summary.csv
│       │   ├── issue_matrix.csv
│       │   └── bug_candidates.csv
│       └── manual_review/                  # 人工真值、截图和复核意见
└── docs/
    └── DFM批量验证计划.md                   # 本文档
```

Git 策略：原始图片、派生图片和 `runs/` 大体积产物默认忽略；`manifest.csv`、汇总表和人工复核结论保留，可作为比赛的可复现实验证据。原始图片仍须在本地备份，并记录来源与许可证。

## 三、图片集设计

第一版固定为 24 张，每组 4 张：

| 组 | 内容 | 主要用途 |
|---|---|---|
| A | 宽底座、低重心、单主体的简单文创手办 | 正向基准，发现 DFM 误报 |
| B | 双脚、单脚、脚尖、上重下轻姿态 | 验证 P1 和摆放类失败 |
| C | 细剑、长角、飘带、镂空灯笼 | 验证壁厚估算及复杂拓扑 |
| D | 平伸手臂、大披风、悬浮配件、多部件 | 验证 S1、G2、G6、D2 |
| E | 环形挂件、中空器皿、拱门、近间隙配件 | 为孔洞、空腔、装配规则建立样本 |
| F | 同一主体的透明、白底、复杂背景、遮挡低对比版本 | 隔离输入处理与生成质量影响 |

图片要求：

1. PNG，建议 1024×1024；主基准图优先 RGBA 透明背景。
2. 主体完整，不能裁掉底部、头顶或细小配件；主体占画面约 70%~85%。
3. 优先三分之四视角，避免多视图拼图、文字、水印、强反光和硬阴影。
4. 除 D 组和 E 组指定样本外，一张图只放一个主体。
5. F 组必须使用同一主体和同一姿态，每张只改变一个输入变量。
6. 文件名必须与 `manifest.csv` 的 `file_name` 一致。
7. `source` 和 `license` 必须填写；比赛样本优先自制、自拍或明确授权素材。

`expected_risk_codes` 只是图片设计预期，不能直接作为三维网格真值。实际缺陷必须在生成后通过网格检查、外部工具和人工复核共同确认。

## 四、修复前后数据口径

必须分开保存三层对象，禁止只比较原始网格和最终网格：

```text
raw_mesh（归一化生成结果）
    ↓ 仅做毫米缩放与 Z=0 放置
initial_prepared_mesh + initial_report
    ↓ 仅做几何修复
final_prepared_mesh + final_report
    ↓ CuraEngine 通用 profile 仿真
slice_result
```

`raw_mesh → initial_prepared_mesh` 是已知源坐标系到打印坐标系、单位和落台变换，不是修复收益；真正的修复效果必须比较 `initial_prepared_mesh → final_prepared_mesh`。当前 Y-up→Z-up 是 Hunyuan 坐标约定转换，不是基于支撑/悬垂代价的自动朝向搜索。

每次运行必须记录：

- 输入：文件哈希、尺寸、颜色模式、alpha、前景占比
- 推理：模型版本、seed、steps、octree、num_chunks、目标高度
- 性能：生成、检查、修复、切片耗时和峰值显存
- 网格：顶点数、面数、组件数、边界、尺寸、水密、绕序
- 初检：每个检查码的状态、分数、指标、详情和阻断属性
- 修复：动作顺序、阶段进度、每步耗时/成功状态、前后顶点/面数、错误信息，以及 `timed_out/timeout_stage/resume_from_stage`
- 复检：最终报告以及相对初检的 PASS/FAIL/UNKNOWN 变化
- 切片：成功状态、层数、耗时、耗材、刀路边界、错误和警告
- 产物：各阶段网格、JSON 报告、日志和异常堆栈

## 五、批量脚本设计要求

已实现 `scripts/batch_dfm_benchmark.py` 的 **raw mesh 重跑模式**：从既有 `benchmarks/dfm_v1/runs/<image_id>/.../raw_mesh.*` 发现唯一输入，执行初检、修复/复检、精检门禁和 Cura 仿真。它不会重新运行图生模型；后续 24 图正式测试仍需增加 manifest 驱动的图生阶段。

当前可执行命令：

```powershell
& "D:\conda\envs_dirs\agent\python.exe" scripts\batch_dfm_benchmark.py `
  --source-runs-root benchmarks\dfm_v1\runs `
  --output-root benchmarks\dfm_v1\reruns `
  --run-id dfm_full_rerun_20260804 `
  --image-ids A01 B01 C01 D01 E01 `
  --target-height 100 `
  --resume
```

`--resume` 复用已完成阶段；运行中/失败阶段会重跑；已记录超时默认作为可审计终态复用，迁移到更强机器后需要重试时显式增加 `--retry-timeouts`。恢复前会核验 raw mesh SHA-256 和配置指纹，不一致时拒绝混用且不修改旧检查点。

脚本必须满足：

1. raw mesh 重跑按显式 `--image-ids` 顺序执行且拒绝同一 ID 的多输入歧义；正式图生批处理仍须按 `manifest.csv` 驱动。
2. 固定参数和 seed，并把全部配置写入 `run.json`。
3. 每个样本使用独立目录和 `try/except/finally`，单张失败不能中断全批次。
4. 每完成一个阶段立即原子落盘，支持 `--resume` 跳过已完成阶段。
5. 保存初始报告和最终报告，不能只调用覆盖初始报告的高层便捷入口。
6. DFM 未通过时不进入正式导出/切片；诊断网格必须标记 `diagnostic_only`。
7. 所有 `UNKNOWN` 单独统计，不能转成 PASS 或 FAIL。
8. 汇总时同时生成机器可读 CSV/JSON 和答辩图表所需的长表数据。
9. 使用 `repair_stage_timeout_seconds` 限制单个危险修复阶段；超时立即保存最后有效网格与恢复位置，再处理下一个样本。

## 六、执行轮次

### 第一轮：链路调试

- 样本：24 张全部
- 参数：mini turbo、5 steps、octree 128、seed 0
- 目标：发现崩溃、单位错误、报告/网格不一致、误导出和无法恢复任务等代码问题

### 第二轮：随机稳定性

- 样本：从第一轮选择 12 张代表性或问题样本
- 参数：保持其他配置不变，补跑 seed 1、2
- 新增运行：24 次
- 目标：判断缺陷是否随 seed 波动，区分生成模型质量和确定性代码问题

### 第三轮：比赛质量

- 样本：选择 6 张具有代表性的国潮/校园文创图片
- 参数：turbo 25 steps、octree 256、固定 seed
- 流程：生成 → 初检 → 修复 → 复检 → Cura 仿真 → 人工复核
- 目标：形成可放入答辩材料的修复前后对比和量化证据

预计总运行数为 54 次，不需要把 24 张图片全部跑 3 个 seed。完整耗时须由第一轮批量脚本实测，不能按单张冒烟时间直接承诺。

## 七、汇总指标

至少输出：

- 修复前快速层 PASS / FAIL / INCOMPLETE 数量和比例
- 修复后快速层 PASS / FAIL / INCOMPLETE 数量和比例
- 每个检查码的失败率和 UNKNOWN 率
- 各检查码修复恢复率：初检 FAIL → 复检 PASS
- 修复回归率：初检 PASS → 复检 FAIL
- DFM PASS 后的 Cura 仿真成功率
- 分组、seed、输入背景和模型参数对应的失败分布
- 生成、检查、修复、切片耗时及峰值显存分布

总分变化必须和报告完整性同时展示，禁止把 `INCOMPLETE: 100` 表述为全部通过。

## 八、问题分类规则

Bug 清单分为四类：

| 分类 | 典型证据 |
|---|---|
| `CODE_BUG` | 崩溃、NaN、同一网格重复检查结果变化、报告与网格不一致、修复引入无关缺陷、外部工具与代码明显矛盾 |
| `MODEL_QUALITY` | 缺陷已存在于 raw_mesh，在 MeshLab/Cura 中可见，且检测在相同网格上可重复 |
| `PROCESS_CONFIG` | 改变目标尺寸、喷嘴、材料或阈值后结论发生符合工艺逻辑的变化 |
| `NOT_IMPLEMENTED` | 当前规则明确返回 UNKNOWN，属于功能待办而不是代码 Bug 或模型失败 |

每条问题记录：`bug_id`、样本和 run、检查码、分类、严重级别、复现步骤、证据文件、预期结果、实际结果、临时处理和状态。

## 九、后续操作清单

- [x] 创建数据集目录和 6 组图片目录
- [x] 创建运行产物、汇总和人工复核目录
- [x] 创建 24 张图片的 `manifest.csv` 模板
- [x] 配置大体积原图、派生图和逐次运行产物的 Git 忽略规则
- [x] 底座自动生成(`pedestal.py`)：P1/P3 失败时自动加底座 + 多级布尔融合
- [x] 碎片自动清理(`fragment_cleaner.py`)：G6 失败时自动删悬浮碎面 + AABB/距离保护
- [x] P3 静态稳定性检测：重心投影与支撑凸包的安全裕量判定
- [x] 修复闭环多轮重试编排：底座→碎片→拓扑修复→追加清理→底座重试（每类 ≤2 次）
- [x] 候选网格事务验收：FAIL 数量不增加、分数不下降且不新增 UNKNOWN 才提交；退化候选回滚并停止派生修复
- [x] 修复阶段硬超时与恢复信息：独立工作进程、结构化进度、最后有效网格、超时/续跑阶段字段
- [x] 使用 A01/E01 已落盘 raw mesh 回归坐标标准化和超时恢复
- [x] A01~E01 五张透明背景图完成 mini 冒烟生成与初检
- [x] 开发 `scripts/batch_dfm_benchmark.py` 的 raw mesh 可恢复模式：阶段原子落盘、配置/输入指纹、`--resume`、`--retry-timeouts`、单样本失败隔离、CSV/JSON 汇总
- [x] 为批量脚本补充 6 项无模型权重回归测试，并增加 B01 修复回滚测试；项目全量 87 项测试通过
- [x] A01~E01 复用 raw mesh 完整重跑：3 个 COMPLETED、2 个安全超时、0 个样本级 FAILED；E01 快速层 PASS 并成功完成 Cura 仿真
- [x] 生成第一版重跑记录与 Bug 分类：`benchmarks/dfm_v1/summaries/dfm_full_rerun_20260804.md`
- [x] B01 事务回滚真实验证：最终保持 P1/G5/G6、70 分，退化候选未提交，回滚后未继续中度/追加底座；结果位于 `benchmarks/dfm_v1/reruns/dfm_b01_transactional_v2_20260804/`
- [ ] 将 24 张图片放入对应目录并补全来源、许可证和备注
- [ ] 编写清单校验：文件存在、文件名唯一、图片可解码、尺寸/模式合法、哈希不重复
- [ ] 扩展批量脚本的 manifest 驱动图生模式，并把生成阶段也纳入原子落盘/恢复
- [ ] 执行第一轮 24 张 smoke 测试
- [ ] 人工可视化复核 A01/B01 的大量连通分量以及 E01 的 W1 薄壁位置
- [ ] 修复代码问题后原样重跑，形成回归对比
- [ ] 选择 12 张执行多 seed 稳定性测试
- [ ] 选择 6 张执行比赛质量与 Cura 仿真测试
- [ ] 汇总前后通过率、修复恢复率、回归率和切片成功率

## 十、阶段完成标准

只有同时满足以下条件，才把“DFM 验证完成”写入项目进度：

1. 24 张第一轮均有独立 `run.json`，失败任务也有结构化错误记录。
2. 初检、修复、复检数据可通过同一 `image_id/run_id` 关联。
3. 所有非 PASS 导出都明确标记为诊断产物。
4. Bug 清单中的代码问题有复现用例，修复后有回归测试。
5. 模型质量、工艺配置和未实现能力没有混入代码 Bug 数量。
6. 汇总结果可由脚本从逐次 JSON 重新生成，而不是人工填写。
