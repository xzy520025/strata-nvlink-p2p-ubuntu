# Strata v0.1.45：Ubuntu 双 2080 Ti P2P 一键导入

本包把已测试的专家 P2P、RAM 所有权去重和批处理 fast-join 源码带到另一台已装 Strata 的 Ubuntu 机器，按目标机器编译后切换指定 JSON 配置。安装期间旧服务继续运行，安装器不会停止或重启服务。

## 使用条件

- Ubuntu x86_64，已验证 Ubuntu 24.04；两张 RTX 2080 Ti（SM75），驱动正常且支持双向 P2P。脚本会报告实际互连，P2P 可用不等于一定通过 NVLink。
- 目标机器已安装 Strata 的 Python 服务，服务具备 `args`、`env`、`exe` 和 Peer 参数透传支持（本机验证的服务为 v0.1.44.1，搭配 v0.1.45 引擎）。
- 原模型配置能正常运行，具有该模型自己的 `--expert-profile`、启用的专家缓存，以及 `--resident-budget-gib` 或 `--resident-experts`。
- CMake ≥ 3.24、Ninja、Git、C/C++ 编译器、Python 3、CUDA Toolkit。优先使用 CUDA 12.8；驱动中的 CUDA version 不代表已经安装 `nvcc`。不自动安装或替换驱动和 Toolkit。

如缺少基础工具，在 Ubuntu 24.04 运行：

```bash
sudo apt update
sudo apt install build-essential cmake ninja-build git python3
```

源码和所需 ggml 已随包提供，编译不需要 GitHub 下载或模型下载。保留源码许可证和 `SOURCES.json` 来源清单；`payload/ggml-files.json` 记录实际 ggml 文件哈希。本包包含的 Strata 固定为 v0.1.45，而非随安装时的最新分支变化。

## 解压与一键导入

将压缩包和 `.sha256` 文件放在同一目录。以下路径请改为目标机器自己的 Strata 安装目录和配置：

```bash
sha256sum -c strata-p2p-v0.1.45-ubuntu-sm75-source.tar.gz.sha256
tar -xzf strata-p2p-v0.1.45-ubuntu-sm75-source.tar.gz
bash ./strata-p2p-v0.1.45/install.sh --strata-dir "$HOME/strata" --config "$HOME/strata/strata-iq3_s.json"
```

无需用 sudo 运行安装器。`--config` 使用完整路径，可位于安装目录之外。路径可含中文和空格，使用引号。

安装顺序为：文件校验 → 配置/工具/双卡检查 → 解压并应用补丁 → Release/SM75 编译 → 两个 GPU 小型正确性测试 → 备份并原子替换配置。默认两路编译、低优先级。测试返回“跳过”会停止安装，不当作通过。

先做只读预检：

```bash
bash ./strata-p2p-v0.1.45/install.sh --strata-dir "$HOME/strata" --config "$HOME/strata/strata-iq3_s.json" --dry-run
```

指定编译器、GPU、并行编译数，或补充原配置缺少的目标模型 profile/RAM 预算：

```bash
bash ./strata-p2p-v0.1.45/install.sh \
  --strata-dir "$HOME/strata" --config "$HOME/strata/strata-iq3_s.json" \
  --nvcc /usr/local/cuda-12.8/bin/nvcc --gpus 0,1 --jobs 2 \
  --expert-profile "/path/to/target-model-profile.bin" --resident-budget-gib 40
```

`40` 是本机 64 GiB 内存的现有专家预算示例，不适用于所有内存容量。参数未提供时复用目标配置原有预算；不复制制作机器的 profile 或模型。物理 GPU 编号取自 `nvidia-smi`；默认复用原配置的双卡顺序，否则为 `0,1`。引擎中的 Peer 始终是这个列表的第二张卡。

## 安装结果

- 完整导入源码、构建目录和日志放在目标 Strata 的 `local-p2p/<包标识>/`。
- 新引擎和配套 `libstrata_kq.so` 放在独立的 `engine/p2p-v0.1.45-sm75-<包标识>/`；原引擎保留。
- 指定 JSON 的 `exe` 指向新引擎；启用 `--peer-device 1`、`--peer-ram-dedup`、`--peer-decode-p2p`、`--peer-decode-fast-join`。
- Peer 预留显存默认 1536 MiB，专家槽上限默认 `0`（自动填充），已有显式值保留。删除冲突的分层、阶段裁剪、副卡 helper 缓存选项。
- 添加 `STRATA_BF16_TC=1`、`STRATA_SM75_PERF=1`、`STRATA_XN_F16=1`、`STRATA_SELECT_SIMT=1`；其余环境变量保留。
- 模型路径、上下文、并发、MTP、视觉、采样、端口和 API key 继续使用目标机器的配置。设置监听局域网却没有 API key 时，先停止导入，要求补齐现有配置。
- 配置和备份文件权限为 0600，备份目录为 0700。编译/测试失败不替换配置；配置中途变化时停止覆盖。

配置只在下次启动时生效。安装完成后，先用目标机器原来的方法停止旧服务，再复制安装器打印的启动命令。如果原服务由 systemd 管理，请手动重启原来的服务单元，避免同时启动两个占用相同端口和显存的实例。

## 回滚

安装完成时会打印准确的备份目录与回滚命令，例如：

```bash
bash ./strata-p2p-v0.1.45/install.sh --rollback "/path/to/strata/local-p2p/backups/20261018-123456-xxxx"
```

回滚恢复原始 JSON 字节和权限；再按原方式重启服务即可恢复原引擎。安装后若 JSON 又被编辑，自动回滚会停止，避免丢失新设置；此时先保存新设置，再对照备份合并。重复安装不会重复添加 Peer 参数，配置相同时不会创建多余备份。

## 验证与适用范围

本工具仅封装既有 CUDA 引擎补丁，没有改写引擎算法。源实现包括 P2P 输入/结果传输、系统栅栏与完成标志、批处理 fast-join、CPU 路径保留和专家 RAM 去重。它按专家协作，不是完整主干张量并行。

安装时运行 `peer_decode_p2p_test` 和 `peer_ram_dedup_test`，检查捕获窗口、字节结果、fast-join、空份额、取消路径及 RAM 所有权交换。它们使用小型合成数据，不加载另一个完整模型。探针采用 `CUDA_MODULE_LOADING=LAZY` 降低显存负担，不改变原服务的环境设置。若旧服务占满显存，测试可能失败；原配置仍保持可用，可在自己方便停止旧服务后重试。

包内附安装器测试，不需要 GPU 或模型，可在解压目录运行：

```bash
python3 tests/test_installer.py
```

覆盖配置迁移、设备 UUID 映射、重复安装、私密备份权限、精确回滚、损坏压缩包、失败保护和配置编辑冲突。安装日志位于 `local-p2p/<包标识>/logs/`；包标识按源码内容生成，安装器包版本另记为 `package_id`，修正文档或安装器不会迫使相同源码从头编译。

这不代替目标模型的真实启动与文本/视觉功能检查，也不保证另一台机器有相同速度。当前方案验证范围为 Ubuntu 24.04、CUDA 12.8、双 22 GiB 2080 Ti 与 NVLink；HIP、SYCL、Windows 和其他显卡没有作为本包验证范围。
