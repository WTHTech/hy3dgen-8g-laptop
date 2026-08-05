# AI 底座图像预处理功能规格

## 目标

在图生 3D 前识别需要实体底座的手办图片，使用带蒙版的 AI 图像编辑在人物底部添加简单、对称、可识别厚度的底座，再把处理后的透明 PNG 交给 Hunyuan3D。该步骤用于降低 P1（无有效平台接触面）的发生率，不能代替后续 DFM 检查。

用户可以选择 `auto`、`always` 或 `never`：

- `auto`：仅手办类图片且底部接触风险高时编辑。
- `always`：强制尝试添加底座。
- `never`：保持原图，不调用外部服务。

## 技术栈

- Python 3.10
- Pillow：RGBA 检查、主体包围框和底部蒙版生成
- OpenAI Python SDK 2.36.0
- OpenAI `gpt-image-2`：带输入图和蒙版的图像编辑
- 火山方舟 Seedream 5.0 Pro：单图参考编辑，独立官方提供方
- 现有 `scripts/inference.py`：Hunyuan3D 图生 3D
- `unittest`：单元和轻量集成测试

## 命令

聚焦测试：

```powershell
& "D:\conda\envs_dirs\agent\python.exe" -m unittest tests.test_image_preprocess -v
```

全量测试：

```powershell
& "D:\conda\envs_dirs\agent\python.exe" -m unittest discover -s tests -p "test_*.py"
```

真实 API 验证必须显式开启，默认测试不会发送图片或消耗额度：

```powershell
& "D:\conda\envs_dirs\agent\python.exe" scripts/preprocess_image.py `
  data/images/A01_baseline_kimono_chibi_girl_rgba.png `
  --mode always --provider seedream `
  --model doubao-seedream-5-0-pro-260628 `
  --output-dir output/preprocessed/A01_seedream
```

## 项目结构

```text
image_preprocess/
├── __init__.py             # 公共类型和入口
├── analyzer.py             # RGBA 主体边界与底部接触风险
├── mask.py                 # 仅允许编辑底部区域的透明蒙版
├── pipeline.py             # 触发、调用、校验、回退与清单
├── prompts.py              # 可制造性底座约束提示词
└── providers/
    ├── __init__.py
    ├── base.py             # 提供方协议与结果类型
    ├── openai_image.py     # OpenAI 图像编辑适配器
    └── volcengine_seedream.py # 火山方舟 Seedream 图像编辑适配器
scripts/preprocess_image.py # 可复用命令行入口
tests/test_image_preprocess.py
```

## 代码风格

提供方边界使用显式类型，外部结果在进入流水线前完成校验：

```python
class ImageEditProvider(Protocol):
    def add_pedestal(
        self,
        image: Image.Image,
        mask: Image.Image,
        request: PedestalEditRequest,
    ) -> ImageEditResult:
        ...
```

- 公共结果使用 `dataclass`。
- 文件写入使用临时文件加原子替换。
- 不记录 API Key、请求头或完整异常响应。
- 外部 API 失败必须返回结构化失败，不能静默伪装为成功。

## 测试策略

- 单元测试：主体分析、触发策略、蒙版范围、透明 PNG 校验。
- 提供方测试：使用假客户端验证参数和响应解析，不访问网络。
- 流水线测试：成功保存派生图；失败回退原图并保存原因；`never` 不调用提供方。
- 接入测试：Hunyuan3D 入口接收预处理后的路径，但使用假生成器避免加载模型。
- 真实验证：仅 A01 一张图片，需显式命令和已配置的 API Key。

## 边界

### 始终执行

- 保留用户原图，派生图写入独立目录。
- 保存 mask 和不含密钥的 `preprocess_manifest.json`。
- 验证输出是有效 RGBA PNG、尺寸合理且主体没有消失。
- API 超时或失败时回退原图，不阻断图生 3D。
- 图生 3D 后仍运行 DFM；底座图仅降低 P1 风险。

### 需要另行确认

- 对多张图片进行真实付费 API 批处理。
- 更换模型、供应商或新增依赖。
- 将用户原图发送到 OpenAI 之外的服务。

### 绝不执行

- 将 API Key 写入源码、配置、日志、测试快照或 Git。
- 对吊坠、挂件等非落地作品强制添加底座。
- 因外部服务失败覆盖或删除用户原图。
- 把 AI 生成成功等同于 DFM 或可打印验证通过。

## 成功标准

1. `auto/always/never` 三种策略行为可测试且确定。
2. `OPENAI_API_KEY` 缺失时给出安全错误，日志不泄露任何密钥信息。
3. API 请求使用 `gpt-image-2`、输入图、底部蒙版、透明 PNG 输出。
4. 输出无效、主体消失或 API 异常时自动回退原图，并记录机器可读原因。
5. 原图、派生图、mask、manifest 可追溯且使用 SHA-256 标识。
6. 全量测试不回归，新增测试全部通过。
7. A01 能完成一次受控真实编辑，人工确认人物主体基本不变且底座清晰连接。

## 已确认决策

- 默认外部提供方为 OpenAI `gpt-image-2`。
- OpenAI 提供方固定使用官方 `https://api.openai.com/v1`，不继承全局
  `OPENAI_BASE_URL`；第三方兼容 API 必须作为独立提供方显式接入。
- Seedream 提供方仅允许火山方舟官方 HTTPS 端点，默认模型为
  `doubao-seedream-5-0-pro-260628`；优先读取 `ARK_API_KEY`，兼容当前
  `OPENAI_API_KEY`，地址读取 `ARK_BASE_URL` 或当前 `BASE_URL`。
- Seedream 没有独立 mask 参数，使用已经留出底部空间的编辑画布作为单张参考图；
  输出仍必须通过透明背景与主体存在性验收，否则回退原图。
- 当前阶段只实现手办底座，不扩展吊坠挂环。
- 默认质量模式优先保持输入人物，编辑范围限制在底部蒙版。
- 外部失败采用原图回退；后续可增加本地确定性底座提供方。

## 2026-08-05 真实验证记录

- 首次 A01 请求状态：`FALLBACK / provider_error`。
- 根因：环境存在非官方 `OPENAI_BASE_URL`，旧实现隐式继承了该端点。
- 修复：官方提供方固定 OpenAI 官方端点，并增加脱敏状态分类和不透明背景回退。
- 首次请求证据：`output/preprocessed/A01_openai_20260805/`。
- 官方端点验证：用户明确授权后已执行一次，结果为
  `FALLBACK / authentication_error`，未生成派生图且没有重试。
- 结论：当前 `OPENAI_API_KEY` 无法通过 OpenAI 官方端点认证；在更换为可用的
  OpenAI 官方密钥，或另行确认并实现第三方兼容提供方前，不再发送图片。
- 官方验证证据：`output/preprocessed/A01_openai_official_20260805/`。

## Seedream 5.0 Pro 接入决策

- 用户已明确授权将 A01 发送至火山方舟并接受一次图片生成费用。
- 账户模型列表已确认 `doubao-seedream-5-0-pro-260628` 可用。
- 仅执行一张 A01，禁止自动重试和批量调用；不在本次验证中生成 3D。

## Seedream 真实验证记录

- 第一次 A01 调用：`FALLBACK / invalid_request`，未生成派生图、未自动重试。
- 根因：Seedream 5.0 Pro 单图路径不支持 4.x 使用的
  `sequential_image_generation`，且不应显式发送 `stream`。
- 修复：Pro 请求体移除上述两个字段；新增回归测试并完成 104 项全量回归。
- 失败证据：`output/preprocessed/A01_seedream_20260805/`。
- 修复后真实验证尚未执行，需要用户明确同意第二次 A01 调用。
