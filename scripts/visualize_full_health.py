# scripts/visualize_full_health.py
"""
Generates full_health_check.png from outputs/health_report.json
Adapted from rsnn_restructured_C.
"""
import json
import matplotlib.pyplot as plt
from pathlib import Path
import sys # 修正: sysをインポート

# プロジェクトルート基準で health_report.json を読み込む
# (このスクリプトは scripts/ にある想定)
ROOT = Path(__file__).parent.parent
health_file = ROOT / "outputs" / "health_report.json"
output_file = ROOT / "outputs" / "full_health_check.png"

if not health_file.exists():
    print(f"Health report not found. Run 'python tools/health_check.py' first.")
    print(f"Expected at: {health_file}")
    sys.exit(1)

with open(health_file) as f:
    data = json.load(f)

print(f"Visualizing health report from {health_file}")

# --- 1. パッケージ状態 ---
packages = data.get("packages", {})
# requirements.txt で "required": true になっているものだけ可視化 (オプション)
# pkg_to_show = {p: v for p, v in packages.items() if v.get("required", True)}
pkg_to_show = packages # すべて表示
package_names = list(pkg_to_show.keys())
package_status = [1 if pkg_to_show[p]["ok"] else 0 for p in package_names]

# --- 2. スモークテスト ---
smoke_tests = data.get("smoke_tests", {})
smoke_names = list(smoke_tests.keys())
smoke_status = [1 if smoke_tests[s] else 0 for s in smoke_names]

# --- 3. 実験実行 ---
experiments = data.get("experiments", {})
exp_names = list(experiments.keys())
exp_status = [1 if experiments[e].get("ran", False) else 0 for e in exp_names]

# 可視化
fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
fig.suptitle(f"RSNN (DI+LC) Full Health Check\n({data.get('timestamp')})", fontsize=16)

# パッケージ
if package_names:
    axes[0].bar(package_names, package_status, color=["green" if s else "red" for s in package_status])
    axes[0].set_ylim(0, 1.2)
    axes[0].set_ylabel("Package OK")
    axes[0].set_title("Python Packages")
else:
    axes[0].text(0.5, 0.5, "No package data", transform=axes[0].transAxes, ha='center')
    axes[0].set_title("Python Packages")

# スモークテスト
if smoke_names:
    axes[1].bar(smoke_names, smoke_status, color=["green" if s else "red" for s in smoke_status])
    axes[1].set_ylim(0, 1.2)
    axes[1].set_ylabel("Passed")
    axes[1].set_title("Smoke Tests (DI Container)")
else:
    axes[1].text(0.5, 0.5, "No smoke test data", transform=axes[1].transAxes, ha='center')
    axes[1].set_title("Smoke Tests (DI Container)")

# 実験実行
if exp_names:
    axes[2].bar(exp_names, exp_status, color=["green" if s else "red" for s in exp_status])
    axes[2].set_ylim(0, 1.2)
    axes[2].set_ylabel("Ran")
    axes[2].set_title("Experiments (src/main.py)")
else:
    axes[2].text(0.5, 0.5, "No experiment data", transform=axes[2].transAxes, ha='center')
    axes[2].set_title("Experiments (src/main.py)")


# 保存
plt.savefig(output_file)
print(f"Saved visualization to: {output_file}")
# plt.show()
