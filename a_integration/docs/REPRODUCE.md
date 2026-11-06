# 运行与复现说明

## 推荐环境
- Windows 10/11 或 Linux；Python 3.11（按 B 交付说明）。
- 依赖以 `runtime/b_code/requirements-lock.txt` 为准。
- B 的 SMPL 源码随包位于 `runtime/b_code/vendor/smpl`。

## Windows PowerShell

```powershell
cd <解压目录>\L3_A_Integrated_20261106
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r .\runtime\b_code\requirements-lock.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe .\main.py --offline-preview
.\.venv\Scripts\python.exe .\main.py --seed 1101 --max-steps 60 --output-dir .\logs\acceptance_20261108
```

如果团队已有 B 的 Python 3.11 虚拟环境，可先设置 `$env:L3_PYTHON = "D:\SMPL_Project\member_b_1101\.venv\Scripts\python.exe"`，再运行 `scripts/run_acceptance.ps1`，避免重复安装。

## Linux/macOS

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r runtime/b_code/requirements-lock.txt
python -m unittest discover -s tests -v
python main.py --offline-preview
python main.py --seed 1101 --max-steps 60 --output-dir logs/acceptance_20261108
```

## 运行产物
- `A_events.jsonl`：各子任务、Bid、Assignment、反馈和耗时。
- `B_events.jsonl`：B ExecutionAgent 原始事件。
- `D_reviews.jsonl`：D 审核日志。
- `A_summary.json`：空间边界、任务状态、步数和运行摘要。

正式运行前必须确认 `config.py` 中执行器 ID 为 `ExecutionAgent-01`，并确认 D 的 `demo_mode=False`。不要通过直接调用 `adapter.step()` 绕过 B/D 门控。
