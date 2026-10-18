# Strata NVLink P2P · Ubuntu / 双 RTX 2080 Ti

为已安装 Strata 的 Ubuntu 机器导入专家 P2P、专家 RAM 去重和批处理 fast-join。随包提供固定 v0.1.45 源码、13 文件补丁及所需 ggml，在目标机器编译 **SM75**，备份并替换指定模型 JSON，手动重启后生效。

**要求：** Ubuntu x86_64、双 RTX 2080 Ti、双向 P2P、CMake ≥ 3.24、Ninja、C/C++ 编译器、Python 3、CUDA Toolkit（推荐 12.8）。已经具有可运行的 Strata 模型配置及该模型自己的专家 profile / RAM 专家预算。

## 下载并导入

[下载源码导入包与校验文件](https://github.com/xzy520025/strata-nvlink-p2p-ubuntu/releases/tag/v0.1.45-p2p.1)。将以下安装路径改为目标机器自己的目录：

```bash
curl -fLO https://github.com/xzy520025/strata-nvlink-p2p-ubuntu/releases/download/v0.1.45-p2p.1/strata-p2p-v0.1.45-ubuntu-sm75-source.tar.gz
curl -fLO https://github.com/xzy520025/strata-nvlink-p2p-ubuntu/releases/download/v0.1.45-p2p.1/strata-p2p-v0.1.45-ubuntu-sm75-source.tar.gz.sha256
sha256sum -c strata-p2p-v0.1.45-ubuntu-sm75-source.tar.gz.sha256
tar -xzf strata-p2p-v0.1.45-ubuntu-sm75-source.tar.gz
bash ./strata-p2p-v0.1.45/install.sh \
  --strata-dir "$HOME/strata" \
  --config "$HOME/strata/strata-iq3_s.json"
```

也可以克隆本仓库，从 `bundle/` 运行同一个安装器：

```bash
git clone https://github.com/xzy520025/strata-nvlink-p2p-ubuntu.git
cd strata-nvlink-p2p-ubuntu
bash ./bundle/install.sh --strata-dir "$HOME/strata" --config "$HOME/strata/strata-iq3_s.json"
```

编译所需源码已在包内，安装器不下载模型。默认两路、低优先级编译；测试通过后切换配置。目标机器原有模型、上下文、并发、MTP、视觉、端口和 API key 继续使用。安装结束会打印启动与回滚命令；请先结束旧服务，再手动重启。

## 预检、自测和回滚

```bash
bash ./bundle/install.sh --strata-dir "$HOME/strata" --config "$HOME/strata/strata-iq3_s.json" --dry-run
python3 bundle/tests/test_installer.py
bash ./bundle/install.sh --rollback "/path/to/strata/local-p2p/backups/安装器打印的备份目录"
```

[完整中文说明](bundle/README.zh-CN.md)包括依赖、设备选择、指定 nvcc、专家 profile、RAM 预算、日志位置和回滚行为。

## 引擎改动与验证

- [完整引擎补丁：13 个文件](bundle/payload/p2p.patch)，覆盖 P2P 输入/结果传输、GPU 完成同步、CPU 行保留、批处理 fast-join、RAM 所有权与交换。
- [实际验证记录](docs/VALIDATION.zh-CN.md)：Ubuntu 24.04、CUDA 12.8、双 RTX 2080 Ti 22 GiB、NVLink NV2；全新 SM75 编译、18 项安装器测试、两个 GPU 正确性测试、重复安装及精确回滚均通过。
- 安装验证使用隔离配置与小型合成 GPU 数据。目标模型的功能需在手动重启后确认；其他系统、显卡、HIP 和 SYCL 未作为本包验证范围。
- 当前实现按专家协作。NVLink 是物理互连，P2P 是设备访存方式；安装器会记录实际拓扑。

源码标识：`4da3526be58d88a2`。导入包 SHA-256：

```text
4e63de9643cb7bcf5f4fbae7832e78b53098851db51b9fe7efdd49b57f04b9a4
```

## 来源与许可证

基于 [Niko1221/Strata](https://github.com/Niko1221/Strata) v0.1.45；本仓库由 xzy520025 维护。上游、依赖与整合代码采用 MIT，版权和来源见 [LICENSE](LICENSE)、[NOTICE.md](NOTICE.md) 与 [SOURCES.json](bundle/SOURCES.json)。
