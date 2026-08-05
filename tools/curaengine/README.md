# CuraEngine 运行时

仓库不提交完整 Cura 安装目录、安装程序或平台二进制。Windows 本地调试运行时放在：

```text
.cache/tools/curaengine/5.13.0/windows-x64/
```

版本与 SHA256 记录在 `runtime-manifest.json`。`CuraSlicer` 按以下顺序查找：

1. 构造参数 `engine_path` / `resources_path`
2. 环境变量 `CURA_ENGINE_PATH` / `CURA_ENGINE_RESOURCES`
3. 项目 `.cache/tools/curaengine/<version>/<platform>/`
4. 系统 `PATH`

Linux 服务器应在 Docker/CI 中构建固定版本，并通过环境变量提供二进制与
`share/cura/resources` 路径。当前通用配置生成的是 DFM 仿真 G-code，不能直接发送给打印机；
正式打印必须换成目标打印机、挤出机、材料和质量配置。
