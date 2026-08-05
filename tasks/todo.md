# AI 底座图像预处理任务

- [x] Task 1：定义请求、结果、分析和蒙版行为
  - Acceptance：支持 `auto/always/never`，底部蒙版不覆盖主体上部。
  - Verify：`tests.test_image_preprocess` 分析与蒙版测试。
  - Files：`image_preprocess/*.py`、`tests/test_image_preprocess.py`

- [x] Task 2：实现 OpenAI 图像编辑提供方
  - Acceptance：使用环境变量和 `gpt-image-2`；解析 PNG；错误不泄密。
  - Verify：假客户端请求参数、响应和异常测试。
  - Files：`image_preprocess/providers/*.py`、`tests/test_image_preprocess.py`

- [x] Task 3：实现可恢复预处理流水线
  - Acceptance：保存原图、mask、派生图、manifest；失败回退原图。
  - Verify：临时目录中的成功、失败和 `never` 测试。
  - Files：`image_preprocess/pipeline.py`、`tests/test_image_preprocess.py`

- [x] Task 4：增加 CLI 与 Hunyuan3D 可选接入
  - Acceptance：CLI 可单独预处理；现有无 DFM 路径默认行为不变。
  - Verify：CLI 参数测试和 `test_inference_lightweight.py`。
  - Files：`scripts/preprocess_image.py`、`scripts/inference.py`、相关测试

- [ ] Task 5：全量验证与单张真实 API 验证
  - Acceptance：全量测试通过；A01 生成底座图和清单；不生成 3D、不进入批量调用。
  - Verify：全量 unittest、文件校验和人工查看 A01 派生图。
  - Files：测试输出仅写 `output/preprocessed/A01`
  - Status：OpenAI 官方密钥不可用；现已改用独立 Seedream 提供方。104 项全量
    测试通过，A01 派生图验收转由 Task 7 跟踪。

- [x] Task 6：接入火山方舟 Seedream 5.0 Pro
  - Acceptance：独立官方端点、单图输入、单图输出、错误脱敏、失败回退。
  - Verify：假 HTTP 客户端测试、CLI 测试、全量 unittest。
  - Files：`image_preprocess/providers/volcengine_seedream.py`、CLI 与测试。

- [ ] Task 7：A01 Seedream 单张真实验证
  - Acceptance：只调用一次，不生成 3D；保存原图、画布、mask、派生图和 manifest。
  - Verify：人工检查主体保持、底座连接和透明背景。
  - Files：`output/preprocessed/A01_seedream_20260805/`。
  - Status：首次请求因 5.0 Pro 不支持 4.x 组图参数而返回 `invalid_request`；
    参数已修复，104 项测试通过，等待用户授权第二次真实调用。
