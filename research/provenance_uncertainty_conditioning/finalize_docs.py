"""Write final status, review inventory and artifact hashes after task-scope audit."""
from core import *
import subprocess,datetime


def main():
    assert json.loads((OUT/'DELIVERY_AUDIT.json').read_text())['passed']
    assert not (OUT/'ARTIFACT_SHA256.json').exists()
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True)
    worktree=subprocess.check_output(['git','worktree','list'],cwd=ROOT,text=True)
    actual_remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/dynamics'],cwd=ROOT,text=True).strip()
    v=json.loads((OUT/'VERIFICATION.json').read_text());cost=json.loads((OUT/'RESOURCE_ACCOUNTING.json').read_text())
    text=f'''# STATUS — 生成历史来源与不确定性条件化

这是提交前完成快照：{datetime.datetime.now(datetime.timezone.utc).isoformat()}。实验执行/核对HEAD=`{head}`；branch=`{branch}`。最终交付提交号由包含本文件的Git提交及交付消息记录，push后另查询实际远端HEAD验证。

## 位置与Git状态

仓库：`{ROOT}`。代码`research/provenance_uncertainty_conditioning/`；有效结果为本目录；有效大缓存`{CACHE}`。旧原生缓存只读，复用manifest保存准确绝对路径与hash。只有一个worktree：

```text
{worktree}```

提交前远端实际查询：`{actual_remote}`。本轮新目录尚未提交；其他用户未跟踪文件保留，已跟踪旧文件没有修改。没有reset、clean、强制切分支或强推。

## 完成分类

| 分类 | 核对结果 |
| --- | --- |
| 已实现但未运行 | 无固定实验待运行项 |
| 已运行但未完成统计 | 无 |
| 已完成并通过复核 | 有效九次1000步训练、45权重、18选择、528评价组、全部指标/轨迹配对统计、VERIFICATION及DELIVERY_AUDIT |
| 因数据/缓存/算力阻碍 | 无 |
| 已完成尚未提交/远端未出现 | 本快照时是；下一步仅提交本轮两个目录并按任务末尾明确授权正常push dynamics，远端核对在交付消息中记录 |

有效训练严格k2/5/10，共9000更新；三臂/三种子抽样和32成员噪声完全一致。直接k0只做预检、开发保护及正式评价。所有科学/保护选择均为step0，属于负实验结果，不等于执行未完成。

## 关键证据文件

- TASK.md、PROTOCOL.md、config.json、TRAINING_FROZEN.json：任务全文与训练前固定条件/hash。
- PREFLIGHT.json、tests.log、preflight_replay.csv、EVALUATION_TESTS.json、PLAN_CHECKS.json：零初始化、完整CRPS两项/梯度、四步导数、来源/std、future污染、成员与物理时间检查。
- TRAINING_COMPLETE.json、TRAINING_SCOPE_AUDIT.json、training.csv、九个train_C_*.jsonl、weights/45小权重：九次各1000有效更新，未改主干，未加loss/候选或搜参。
- development.csv、development_nonzero_*、development_tradeoffs.png：全部36非零候选及其CRPS/ES/MSE/SMSE/离散度代价，没有用回退隐藏结果。
- selection_candidates.csv、selection.json、FROZEN_SELECTION.json：分别科学/保护规则、全部45候选及18独立选择，冻结早于测试。
- evaluation_plan.json、evaluation_manifest.jsonl、extension_manifest.jsonl、events.jsonl：48历史+480后续组，全部输入/noise/cache/输出hash及先发布后truth事件。
- scores.csv.gz、leads_*.csv.gz、scores_by_*、paired_*、lead_summary.csv：种子/轨迹/Re/通道/lead/k/窗口/repeat/政策/视图结果，原轨迹配对bootstrap。temporal大缓存保留ACF和协方差/相关矩阵。
- ANALYSIS.json、DECISIONS.json、VERIFICATION.json、DELIVERY_AUDIT.json及日志：统计、固定稳定性判定、数值/身份和任务范围复核。
- DECISION_REPORT.md：中文研究报告，明确回答八个问题，区分执行/收益/机制/未识别/独立验证边界。
- RESOURCE_ACCOUNTING.json、EXECUTED_COMMANDS.json：有效/作废批次成本、实际命令；README提供全新输出目录复跑入口。

审计量：{v['groups']}组身份、{v['all_lead_rows_checked']}个lead明细、{v['audit_score_rows']}固定子集评分复算，评分最大差{v['audit_score_maxabs']:.4g}；{v['crps_bootstrap_intervals_checked']}个CRPS轨迹区间重算。

## 作废批次与成本

invalid_direct_training/完整保留错误加入k0训练的先前9000更新、45权重、协议/源码/开发结果及中止评价，明确无效且不进入任何有效选择/统计。PROTOCOL_CORRECTION.md解释只恢复原任务k约束，没有改变架构、lr、训练步数、种子、loss和选择门槛。该批冻结原生extension可按锁定身份hash复用，错误适配器权重和分数不复用。

有效训练/开发{cost['valid_training_seconds']:.3f}秒，正式评价{cost['valid_formal_evaluation_seconds']:.3f}秒；作废训练{cost['invalid_training_seconds']:.3f}秒及中止评价成本单列。大缓存总留存{cost['total_retained_new_cache_bytes']/2**30:.3f}GiB，均不进Git。不是单次四步预测的同预算改进：重启仍需父+子共八次网络调用。

## 剩余工作与限制

固定清单、统计、报告和审计无剩余项；提交前只剩Git提交、正常push和远端HEAD核对。本轮不启动下一研究阶段。

经验结果为未获得可选择的开发CRPS增益；正式回退保留原生S/B，来源/不确定性增量机制未证实。历史测试、后续时段及纠错前探索都已使用过，只有三个测试Re组，不能称新盲测。新的独立轨迹/Re仍是未来研究缺口，不阻碍本轮负结果交付。

## 提交前工作区原样记录

```text
{status}```
'''
    (OUT/'STATUS.md').write_text(text)
    review='''# 审查顺序与文件清单

先读DECISION_REPORT.md、STATUS.md、PROTOCOL_CORRECTION.md；再看PROTOCOL.md/config.json、PREFLIGHT/TRAINING_FROZEN、九训练日志与45权重、全部development候选、selection/FROZEN_SELECTION、EVALUATION_COMPLETE、VERIFICATION/DELIVERY_AUDIT。

有效统计入口：scores_summary.csv、scores_by_trajectory_seed.csv、paired_intervals.csv/paired_intervals_by_Re.csv、lead_summary.csv。原始明细保留在scores.csv.gz和leads_*.csv.gz。方差、宽度或接缝幅度下降不自动等于改善。每个seed及mean3独立列示，但物理统计单位始终是12条原轨迹。

invalid_direct_training/仅作透明审计，禁止混入有效结果；其中SHA256.json锁定所有作废产物。大缓存位于Workspace/runs，Git只收小适配器权重、协议、代码、日志、压缩明细及统计。ARTIFACT_SHA256.json覆盖所有交付文件（不自引用）。

| 文件 | bytes |
| --- | ---: |
'''
    code=Path(__file__).resolve().parent
    for p in sorted(list(code.rglob('*'))+list(OUT.rglob('*'))):
        if p.is_file() and p.name not in ['REVIEW_FILES.md','ARTIFACT_SHA256.json']:
            review+=f'| `{p.relative_to(ROOT)}` | {p.stat().st_size} |\n'
    (OUT/'REVIEW_FILES.md').write_text(review)
    files={str(p.relative_to(ROOT)):digest(p) for p in sorted(list(code.rglob('*'))+list(OUT.rglob('*'))) if p.is_file() and p.name!='ARTIFACT_SHA256.json'}
    dump(OUT/'ARTIFACT_SHA256.json',files)
    # Remain silent: an output log inside OUT is itself included in the inventory.
    # Printing after hashing that file would invalidate its recorded digest.

if __name__=='__main__':main()
