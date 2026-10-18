"""Import the frozen CUDA P2P source build into an existing Strata installation.

Only Python's standard library is used. No service is stopped or started here.
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import fcntl
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path


class InstallError(RuntimeError):
    pass


PERF_ENV = {name: "1" for name in (
    "STRATA_BF16_TC", "STRATA_SM75_PERF", "STRATA_XN_F16", "STRATA_SELECT_SIMT",
)}
PEER_SWITCHES = ("--peer-ram-dedup", "--peer-decode-p2p", "--peer-decode-fast-join")
VALUE_FLAGS = {
    "--layer-split", "--expert-cache-device1", "--expert-cache-device2", "--expert-cache-device3",
    "--expert-cache-remote-placement", "--peer-device", "--peer-reserve-mib", "--peer-slots",
    "--expert-profile", "--expert-cache", "--resident-budget-gib",
}
REMOVE_FLAGS = {
    "--layer-split", "--trim-stage-weights", "--split-skip-if-fits",
    "--expert-cache-device1", "--expert-cache-device2", "--expert-cache-device3",
    "--remote-expert-opt", "--expert-cache-remote-placement",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> dict:
    try:
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as e:
        raise InstallError(f"无法读取 JSON：{path}（{type(e).__name__}）") from e
    if not isinstance(obj, dict):
        raise InstallError(f"JSON 顶层必须是对象：{path}")
    return obj


def atomic_write(path: Path, data: bytes, mode: int = 0o600):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".p2p-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            os.fchmod(f.fileno(), mode)
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json(path: Path, data: dict):
    atomic_write(path, (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode())


def safe_extract(archive: Path, destination: Path):
    """Accept regular files/directories only; never follow archive links."""
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tf:
        members = tf.getmembers()
        for item in members:
            name = Path(item.name)
            if name.is_absolute() or ".." in name.parts or not (item.isfile() or item.isdir()):
                raise InstallError(f"压缩包包含不允许的路径或链接：{item.name}")
            target = (destination / name).resolve()
            if not target.is_relative_to(destination):
                raise InstallError("压缩包路径越界")
        # All parents have been checked, and destination is a fresh private directory.
        extra = {"filter": "data"} if sys.version_info >= (3, 12) else {}
        tf.extractall(destination, members=members, **extra)


def verify_payload(package: Path) -> dict:
    manifest = read_json(package / "manifest.json")
    if manifest.get("schema") != 1 or manifest.get("version") != "0.1.45":
        raise InstallError("不支持的源码包清单版本")
    for relative, expected in manifest["payload_sha256"].items():
        p = (package / relative).resolve()
        if not p.is_relative_to(package.resolve()) or not p.is_file() or sha(p) != expected:
            raise InstallError(f"文件 SHA-256 校验失败：{relative}")
    return manifest


def flag_value(args: list[str], flag: str):
    found = []
    for i, token in enumerate(args):
        if token == flag:
            if i + 1 == len(args) or args[i + 1].startswith("--"):
                raise InstallError(f"参数缺少值：{flag}")
            found.append(args[i + 1])
        elif token.startswith(flag + "="):
            found.append(token.split("=", 1)[1])
    if len(found) > 1:
        raise InstallError(f"参数重复，请先整理配置：{flag}")
    return found[0] if found else None


def without_flags(args: list[str], flags: set[str]) -> list[str]:
    result = []
    i = 0
    while i < len(args):
        key = args[i].split("=", 1)[0]
        if key in flags:
            if key in VALUE_FLAGS and "=" not in args[i]:
                if i + 1 == len(args) or args[i + 1].startswith("--"):
                    raise InstallError(f"参数缺少值：{key}")
                i += 1
        else:
            result.append(args[i])
        i += 1
    return result


def select_gpus(cfg: dict, requested: str | None) -> list[int]:
    if requested:
        try:
            value = [int(s) for s in requested.split(",")]
        except ValueError as e:
            raise InstallError("--gpus 格式应为 0,1") from e
    else:
        value = cfg.get("gpu")
        if not isinstance(value, list) or len(value) != 2:
            value = [0, 1]
    if len(value) != 2 or any(type(v) is not int or v < 0 for v in value) or value[0] == value[1]:
        raise InstallError("必须选择两张不同的显卡，例如 --gpus 0,1")
    return value


def validate_config(cfg: dict):
    if not isinstance(cfg.get("args"), list) or not all(isinstance(x, str) for x in cfg["args"]):
        raise InstallError("目标配置必须包含字符串数组 args")
    if not isinstance(cfg.get("exe"), str) or not cfg["exe"]:
        raise InstallError("目标配置缺少 exe 引擎路径")
    if "env" in cfg and not isinstance(cfg["env"], dict):
        raise InstallError("目标配置的 env 必须是对象")
    if "lib_dirs" in cfg and (not isinstance(cfg["lib_dirs"], list) or
                              not all(isinstance(p, str) for p in cfg["lib_dirs"])):
        raise InstallError("目标配置的 lib_dirs 必须是路径数组")
    host = cfg.get("host", "127.0.0.1")
    if host not in ("127.0.0.1", "localhost", "::1") and not str(cfg.get("api_key") or "").strip():
        raise InstallError("目标配置监听局域网地址但没有 API key；请先为现有服务设置 api_key")


def prepare_config(cfg: dict, root: Path, engine: Path, options) -> dict:
    validate_config(cfg)
    new = json.loads(json.dumps(cfg))
    args = cfg["args"]
    profile = options.expert_profile or flag_value(args, "--expert-profile")
    if not profile:
        raise InstallError("P2P 需要目标模型自己的专家 profile；请提供 --expert-profile /path/profile.bin")
    cwd = Path(cfg.get("cwd", str(root))).expanduser()
    if not cwd.is_absolute():
        cwd = root / cwd
    if not cwd.is_dir():
        raise InstallError("目标配置 cwd 目录不存在")
    profile_path = Path(profile).expanduser()
    if not profile_path.is_absolute():
        profile_path = cwd / profile_path
    if not profile_path.is_file():
        raise InstallError("目标模型的专家 profile 文件不存在，请检查 --expert-profile")
    budget = options.resident_budget_gib if options.resident_budget_gib is not None else flag_value(args, "--resident-budget-gib")
    if budget is not None:
        try:
            valid = float(budget) > 0 and float(budget) < float("inf")
        except (ValueError, TypeError):
            valid = False
        if not valid:
            raise InstallError("resident-budget-gib 必须为正数")
    elif "--resident-experts" not in args:
        raise InstallError("RAM 去重需要 RAM 专家配置；请提供 --resident-budget-gib，或保留原 --resident-experts")
    cache = flag_value(args, "--expert-cache") or "auto"
    if cache != "auto":
        try:
            cache_ok = int(cache) > 0
        except ValueError:
            cache_ok = False
        if not cache_ok:
            raise InstallError("P2P 需要启用专家缓存：--expert-cache auto 或正整数")
    reserve = flag_value(args, "--peer-reserve-mib") or "1536"
    slots = flag_value(args, "--peer-slots") or "0"
    for key, value in (("--peer-reserve-mib", reserve), ("--peer-slots", slots)):
        if not value.isdigit():
            raise InstallError(f"{key} 必须为非负整数")
    replace = REMOVE_FLAGS | set(PEER_SWITCHES) | {
        "--peer-device", "--peer-reserve-mib", "--peer-slots", "--expert-cache", "--expert-profile",
    }
    if options.resident_budget_gib is not None:
        replace.add("--resident-budget-gib")
        # --resident-experts is a boolean; the explicit budget selects budgeted residency.
        replace.add("--resident-experts")
    args = without_flags(args, replace)
    args += ["--expert-profile", str(profile_path.resolve()), "--expert-cache", cache,
             "--peer-device", "1", "--peer-reserve-mib", reserve, "--peer-slots", slots,
             *PEER_SWITCHES]
    if options.resident_budget_gib is not None:
        args += ["--resident-budget-gib", str(options.resident_budget_gib)]
    new.update(exe=str(engine), args=args, gpu=select_gpus(cfg, options.gpus))
    if "gpus_asked" in new:
        new["gpus_asked"] = list(new["gpu"])
    new.pop("layer_split", None)
    new.pop("split_skip_if_fits", None)
    new["env"] = {**new.get("env", {}), **PERF_ENV}
    # A versioned engine directory holds its matching optional K-quant library.
    new["lib_dirs"] = list(dict.fromkeys([str(engine.parent), *new.get("lib_dirs", [])]))
    return new


def shell_run(command: list[str], cwd: Path | None = None, env: dict | None = None,
              log: Path | None = None, timeout: int | None = None) -> str:
    if log:
        with log.open("w") as f:
            p = subprocess.run(command, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
        output = log.read_text(errors="replace")
    else:
        p = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
        output = p.stdout + p.stderr
    if p.returncode:
        message = "测试被跳过（不等于通过）" if p.returncode == 77 else f"退出码 {p.returncode}"
        raise InstallError(f"{Path(command[0]).name} 失败：{message}" + (f"；日志：{log}" if log else ""))
    return output


def prerequisite_tools(options) -> str:
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "amd64"):
        raise InstallError("本包面向 Ubuntu x86_64 / 双 RTX 2080 Ti")
    missing = [t for t in ("cmake", "ninja", "git", "c++", "cc", "nvidia-smi") if not shutil.which(t)]
    if missing:
        raise InstallError("缺少工具：" + ", ".join(missing) +
                           "。基础工具安装命令：sudo apt update && sudo apt install build-essential cmake ninja-build git python3；NVIDIA 驱动另行安装")
    version = shell_run(["cmake", "--version"])
    m = re.search(r"cmake version (\d+)\.(\d+)", version)
    if not m or tuple(map(int, m.groups())) < (3, 24):
        raise InstallError("需要 CMake 3.24 或更新；Ubuntu 24.04 的 cmake 包满足要求")
    if options.nvcc:
        nvcc = str(Path(options.nvcc).expanduser().resolve())
    elif Path("/usr/local/cuda-12.8/bin/nvcc").is_file():
        nvcc = "/usr/local/cuda-12.8/bin/nvcc"
    else:
        nvcc = shutil.which("nvcc")
    if not nvcc or not Path(nvcc).is_file():
        raise InstallError("缺少 CUDA Toolkit 编译器，nvidia-smi 正常不代表已安装 nvcc。请安装 CUDA 12.8 Toolkit，或传入 --nvcc /path/to/nvcc")
    nv = shell_run([nvcc, "--version"])
    m = re.search(r"release (\d+)\.(\d+)", nv)
    if not m or int(m.group(1)) < 12:
        raise InstallError("本包要求 CUDA Toolkit 12.x 或更新，已验证 CUDA 12.8")
    supported = shell_run([nvcc, "--list-gpu-code"])
    if "sm_75" not in supported:
        raise InstallError("这个 CUDA Toolkit 不支持 SM75，请使用 CUDA 12.8")
    return nvcc


def gpu_preflight(gpus: list[int]) -> dict:
    out = shell_run(["nvidia-smi", "--query-gpu=index,name,uuid,compute_cap,memory.total,memory.free",
                     "--format=csv,noheader,nounits"])
    rows = {}
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 6:
            rows[int(parts[0])] = dict(zip(("name", "uuid", "compute_cap", "memory_total_mib", "memory_free_mib"), parts[1:]))
    if any(i not in rows or "2080 Ti" not in rows[i]["name"] or rows[i]["compute_cap"] != "7.5" for i in gpus):
        raise InstallError("所选设备必须为两张 SM75 RTX 2080 Ti；用 --gpus 指定 nvidia-smi 中的编号")
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
        if cuda.cuInit(0):
            raise InstallError("CUDA 驱动初始化失败")
        devices = []
        # Enumerate CUDA devices by UUID; CUDA and nvidia-smi need not use the same order.
        class UUID(ctypes.Structure):
            _fields_ = [("bytes", ctypes.c_ubyte * 16)]
        count = ctypes.c_int()
        if cuda.cuDeviceGetCount(ctypes.byref(count)):
            raise InstallError("无法枚举 CUDA 设备")
        get_uuid = getattr(cuda, "cuDeviceGetUuid_v2", None) or cuda.cuDeviceGetUuid
        by_uuid = {}
        for ordinal in range(count.value):
            device, ident = ctypes.c_int(), UUID()
            if cuda.cuDeviceGet(ctypes.byref(device), ordinal) or get_uuid(ctypes.byref(ident), device.value):
                raise InstallError("无法读取 CUDA 设备 UUID")
            by_uuid[bytes(ident.bytes)] = device.value
        for index in gpus:
            raw = bytes.fromhex(rows[index]["uuid"].removeprefix("GPU-").replace("-", ""))
            if raw not in by_uuid:
                raise InstallError("无法按 UUID 选择 CUDA 设备")
            devices.append(by_uuid[raw])
        for a, b in (devices, devices[::-1]):
            can = ctypes.c_int()
            if cuda.cuDeviceCanAccessPeer(ctypes.byref(can), a, b) or not can.value:
                raise InstallError("所选双卡不支持双向 P2P，停止切换")
    except (OSError, AttributeError) as e:
        raise InstallError("无法加载 NVIDIA CUDA 驱动 libcuda.so.1") from e
    topology = shell_run(["nvidia-smi", "topo", "-m"])
    link = None
    for line in topology.splitlines():
        fields = line.split()
        if fields and fields[0] == f"GPU{gpus[0]}" and len(fields) > gpus[1] + 1:
            link = fields[gpus[1] + 1]
    return {"gpus": [{"index": i, **rows[i]} for i in gpus], "bidirectional_p2p": True,
            "interconnect": link or "unknown", "nvlink_detected": bool(link and link.startswith("NV")),
            "topology": topology}


def cuda_library_dirs(nvcc: str) -> list[str]:
    toolkit = Path(nvcc).resolve().parent.parent
    return [str(p) for p in (toolkit / "lib64", toolkit / "targets/x86_64-linux/lib") if p.is_dir()]


def source_tree(package: Path, manifest: dict, base: Path) -> Path:
    source = base / "source"
    if source.exists():
        expected = read_json(base / "source-files.json")
        if all((source / k).is_file() and sha(source / k) == v for k, v in expected.items()):
            return source
        raise InstallError("缓存源码已修改；请保留修改并使用新的安装目录，或恢复源码后重试")
    unpack = Path(tempfile.mkdtemp(prefix="unpack-", dir=base))
    try:
        safe_extract(package / "payload/upstream.tar.gz", unpack / "upstream")
        tree = unpack / "upstream" / manifest["source_root"]
        git_env = {k: v for k, v in os.environ.items() if k not in
                   ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR")}
        git_env["GIT_CEILING_DIRECTORIES"] = str(tree.parent)
        shell_run(["git", "apply", "--check", str(package / "payload/p2p.patch")], cwd=tree, env=git_env)
        shell_run(["git", "apply", str(package / "payload/p2p.patch")], cwd=tree, env=git_env)
        safe_extract(package / "payload/ggml.tar.gz", unpack / "vendor")
        (tree / "third_party/llama.cpp").mkdir(parents=True, exist_ok=True)
        for p in (unpack / "vendor").iterdir():
            shutil.move(str(p), tree / "third_party/llama.cpp" / p.name)
        hashes = {str(p.relative_to(tree)): sha(p) for p in sorted(tree.rglob("*")) if p.is_file()}
        expected = manifest["patched_files_sha256"]
        if any(hashes.get(name) != value for name, value in expected.items()):
            raise InstallError("应用补丁后的源码与已验证版本不符")
        os.replace(tree, source)
        write_json(base / "source-files.json", hashes)
        return source
    finally:
        shutil.rmtree(unpack)


def build_engine(source: Path, base: Path, nvcc: str, jobs: int, hardware: dict) -> Path:
    build = base / "build"
    logs = base / "logs"; logs.mkdir(exist_ok=True)
    env = dict(os.environ)
    # Probe tests only: lazy module loading keeps their small fixtures beside a running model.
    env["CUDA_VISIBLE_DEVICES"] = ",".join(g["uuid"] for g in hardware["gpus"])
    env["CUDA_MODULE_LOADING"] = "LAZY"
    env["LD_LIBRARY_PATH"] = ":".join([*cuda_library_dirs(nvcc), env.get("LD_LIBRARY_PATH", "")])
    print("[3/5] 配置并编译 Release / SM75；详细进度见 " + str(logs / "build.log"), flush=True)
    shell_run(["cmake", "-S", str(source), "-B", str(build), "-G", "Ninja",
               "-DCMAKE_BUILD_TYPE=Release", "-DSTRATA_ENABLE_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=75",
               f"-DCMAKE_CUDA_COMPILER={nvcc}", "-DSTRATA_BUILD_TESTS=ON",
               f"-DSTRATA_GGML_DIR={source / 'third_party/llama.cpp'}"], env=env, log=logs / "configure.log")
    shell_run(["cmake", "--build", str(build), "--parallel", str(jobs), "--target",
               "strata", "peer_decode_p2p_test", "peer_ram_dedup_test"], env=env, log=logs / "build.log")
    print("[4/5] 验证 P2P、fast-join、RAM 所有权与交换", flush=True)
    for test in ("peer_decode_p2p_test", "peer_ram_dedup_test"):
        shell_run([str(build / test)], env=env, log=logs / (test + ".log"), timeout=150)
    ldd = shell_run(["ldd", str(build / "strata")], env=env, log=logs / "ldd.log")
    if "not found" in ldd:
        raise InstallError("新引擎有动态库缺失，详见 ldd.log")
    write_json(base / "build-result.json", {"nvcc": nvcc, "arch": 75, "jobs": jobs,
        "engine_sha256": sha(build / "strata"), "tests": ["peer_decode_p2p_test", "peer_ram_dedup_test"],
        "probe_module_loading": "LAZY", "hardware": hardware})
    return build


def install_files(build: Path, release: Path, manifest: dict):
    release.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".p2p-release-", dir=release.parent))
    try:
        files = [build / "strata", *sorted(build.glob("libstrata*.so*"))]
        if not (build / "libstrata_kq.so").is_file():
            raise InstallError("构建缺少配套 libstrata_kq.so")
        for p in files:
            shutil.copy2(p, staging / p.name)
        (staging / "strata").chmod(0o755)
        write_json(staging / "BUILD.json", {"version": manifest["version"], "bundle_id": manifest["bundle_id"],
            "engine_sha256": sha(staging / "strata"), "libraries": {p.name: sha(p) for p in files[1:]}})
        if release.exists():
            # Immutable versioned release: never overwrite an executable used by a running process.
            before = read_json(release / "BUILD.json")
            after = read_json(staging / "BUILD.json")
            if before != after or any(sha(release / p.name) != sha(p) for p in files):
                raise InstallError("同名版本目录已有不同构建；请保留现场并检查编译环境")
        else:
            os.replace(staging, release)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def commit_config(path: Path, original: bytes, cfg: dict, backup_root: Path, package: Path) -> Path | None:
    if path.read_bytes() != original:
        raise InstallError("编译期间配置已被其他程序修改；未覆盖。请重新运行，已完成的编译会复用")
    if read_json(path) == cfg:
        print("配置已是当前 P2P 版本，无需重复备份或写入。", flush=True)
        return None
    backup_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(backup_root, 0o700)
    backup = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("%Y%m%d-%H%M%S-") , dir=backup_root))
    original_mode = stat.S_IMODE(path.stat().st_mode)
    atomic_write(backup / "config.before.json", original)
    installed = (json.dumps(cfg, ensure_ascii=False, indent=2) + "\n").encode()
    write_json(backup / "rollback.json", {"config": str(path), "before_sha256": digest(original),
        "installed_sha256": digest(installed), "original_mode": original_mode,
        "installer": str(package / "install.sh"), "created_at": datetime.now(timezone.utc).isoformat()})
    if path.read_bytes() != original:
        raise InstallError("写入前配置发生变化；已保留备份，未覆盖配置")
    atomic_write(path, installed)
    return backup


def rollback(backup: Path):
    backup = backup.expanduser().resolve()
    metadata = read_json(backup / "rollback.json")
    config = Path(metadata["config"])
    original = (backup / "config.before.json").read_bytes()
    if digest(original) != metadata["before_sha256"]:
        raise InstallError("回滚备份损坏，未修改配置")
    current = config.read_bytes()
    if digest(current) == metadata["before_sha256"]:
        print("原配置已经恢复，无需重复操作。")
        return
    if digest(current) != metadata["installed_sha256"]:
        raise InstallError("安装后配置已被修改；回滚停止，请先保存这些修改再人工合并备份")
    atomic_write(config, original, metadata["original_mode"])
    print("已恢复原配置；当前服务未重启。配置：" + str(config))


def print_restart(root: Path, config: Path):
    py = root / ".venv/bin/python"
    if not py.is_file():
        py = Path(sys.executable)
    print("\n配置将在下次启动生效。请先用目标机器现有的停止方式结束旧服务，再启动：")
    print(shlex.join([str(py), str(root / "serve/server.py"), "--engine", "strata", "--config", str(config)]))
    print("如果现有服务由 systemd 管理，请在方便时重启它原来的服务单元。")


def parser():
    p = argparse.ArgumentParser(description="Strata v0.1.45 双 2080 Ti P2P 源码导入；替换配置后手动重启")
    p.add_argument("--strata-dir", type=Path)
    p.add_argument("--config", type=Path)
    p.add_argument("--nvcc", help="CUDA Toolkit 的 nvcc 路径，优先自动寻找 CUDA 12.8")
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--gpus", help="nvidia-smi 的两个物理编号，默认保留原双卡列表，否则为 0,1")
    p.add_argument("--expert-profile", help="目标模型自己的专家 profile")
    p.add_argument("--resident-budget-gib", type=float, help="仅在显式指定时替换原 RAM 专家预算")
    p.add_argument("--rollback", type=Path)
    p.add_argument("--dry-run", action="store_true", help="只检查输入、源码包、工具和双向 P2P，不写文件")
    return p


def main(argv=None):
    options = parser().parse_args(argv)
    if options.rollback:
        rollback(options.rollback)
        return
    if not options.strata_dir or not options.config:
        raise InstallError("需要 --strata-dir 和 --config；先解压安装包，再运行其中的 install.sh")
    if options.jobs < 1:
        raise InstallError("--jobs 必须为正整数")
    package = Path(__file__).resolve().parent
    root = options.strata_dir.expanduser().resolve()
    config = options.config.expanduser().resolve()
    server = root / "serve/server.py"
    if not server.is_file():
        raise InstallError("目标目录没有 serve/server.py，请指定已安装 Strata 的目录")
    # This is the interface used by the tested 0.1.44.1 server / 0.1.45 engine pair.
    text = server.read_text(encoding="utf-8-sig")
    if not all(marker in text for marker in ("--peer-device", "def engine_args(", "def layer_split_of(")):
        raise InstallError("目标 Python 服务未提供本包需要的 Peer 参数透传接口；请先升级 Strata 服务")
    original = config.read_bytes()
    cfg = read_json(config)
    print("[1/5] 校验源码、现有配置和工具", flush=True)
    manifest = verify_payload(package)
    release = root / "engine" / ("p2p-v0.1.45-sm75-" + manifest["bundle_id"])
    new = prepare_config(cfg, root, release / "strata", options)
    nvcc = prerequisite_tools(options)
    new["lib_dirs"] = list(dict.fromkeys([*new["lib_dirs"], *cuda_library_dirs(nvcc)]))
    print("[2/5] 核对双 2080 Ti、双向 P2P 和实际拓扑", flush=True)
    hardware = gpu_preflight(new["gpu"])
    print("互连：" + hardware["interconnect"] + ("（检测到 NVLink）" if hardware["nvlink_detected"] else "（不是已确认的 NVLink 链路）"), flush=True)
    if options.dry_run:
        print("dry-run 通过：尚未解压、编译或修改配置。")
        return
    with contextlib.suppress(OSError):
        os.nice(10)
    local = root / "local-p2p"
    local.mkdir(exist_ok=True, mode=0o700)
    with (local / "install.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise InstallError("这个 Strata 目录已有另一个 P2P 安装任务") from e
        base = local / manifest["bundle_id"]; base.mkdir(exist_ok=True)
        source = source_tree(package, manifest, base)
        build = build_engine(source, base, nvcc, options.jobs, hardware)
        install_files(build, release, manifest)
        print("[5/5] 备份并原子替换模型配置", flush=True)
        backup = commit_config(config, original, new, local / "backups", package)
        if backup:
            print("配置备份：" + str(backup))
            print("回滚命令：" + shlex.join(["bash", str(package / "install.sh"), "--rollback", str(backup)]))
        write_json(base / "installation.json", {"completed_at": datetime.now(timezone.utc).isoformat(),
            "config_path": str(config), "engine": str(release / "strata"), "bundle_id": manifest["bundle_id"],
            "package_id": manifest["package_id"],
            "config_sha256": sha(config), "backup": str(backup) if backup else None,
            "service_restarted": False})
    print_restart(root, config)


if __name__ == "__main__":
    try:
        main()
    except (InstallError, OSError, subprocess.SubprocessError) as exc:
        print("安装停止：" + str(exc), file=sys.stderr)
        sys.exit(1)
