"""Create a measured, provenance-aware report after the complete audit."""
import csv
import hashlib
import json
import statistics
from pathlib import Path
from train49 import OUT


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def main():
    assert json.loads((OUT/'independent_audit.json').read_text())['passed']
    rows=list(csv.DictReader((OUT/'summary/macro49.csv').open()))
    selected=[r for r in rows if r['split']=='test' and r['mode']=='online']
    windows=json.loads((OUT/'summary/test_distribution_windows.json').read_text())
    methods=('XPF16','8-noise','32-noise','256-noise')
    lines=['# 49 个独立 XPF16 双分位模型与 8/32/256 样本对照', '',
           '本报告仅在 49 个模型、全部评估和独立审计完成后生成。每个模型输入 16 维特征，只输出两个分位数；不是 98 输出联合网络。四步 DiPred 一次预测完整未来 20 帧。', '',
           '测试集为历史上多次使用过的轨迹；虽然本轮训练、校准、开发和测试按轨迹分离，不能把它称为全新盲测。报告为探索性，覆盖率是逐点边际指标。', '',
           '| 在线方法 | 经验 CRPS | Fair CRPS 统计量 | WIS |', '|---|---:|---:|---:|']
    for m in methods:
        if m=='XPF16':
            crps=statistics.mean(w['online_rearranged_crps'] for w in windows)
            fair='—';wis=statistics.mean(w['online_rearranged_wis'] for w in windows)
            label=f'{crps:.6f}（分位积分近似）'
        else:
            crps=statistics.mean(w[f'online_{m}_empirical'] for w in windows)
            fair=f"{statistics.mean(w[f'online_{m}_fair'] for w in windows):.6f}"
            wis=statistics.mean(w[f'online_{m}_wis'] for w in windows)
            label=f'{crps:.6f}'
        lines.append(f'| {m} | {label} | {fair} | {wis:.6f} |')
    audit=json.loads((OUT/'independent_audit.json').read_text())
    projection=next(x for x in audit['split']['test']['projection'] if x['experiment']=='XPF16-Q49')
    projection_pct=100*projection['points']/projection['total']
    raw_cross=100*statistics.mean(w['raw_unrearranged_crossing_any'] for w in windows)
    online_cross=100*statistics.mean(w['online_unrearranged_crossing_any'] for w in windows)
    lines += ['', f'独立模型拼接后，测试第 2–42 个窗口中，至少发生一次跨模型相邻分位交叉的位置比例：Raw {raw_cross:.2f}%，Online {online_cross:.2f}%；单调重排后为 0%。Q49 的在线上下界在全部 42 个窗口有 {projection_pct:.2f}% 的位置发生交叉并被投影到中点，窄区间结果需要谨慎解释。', '',
              '上述分数按训练集通道标准差归一化；其窗口聚合为轨迹/通道等权，与覆盖汇总的 Re 组/通道等权口径可能略有差异。', '',
              '| 目标覆盖 | 方法 | 实际覆盖 | 平均宽度 | 区间分数 |', '|---:|---|---:|---:|---:|']
    for k in (1,5,10,25,40,49):
        for m in methods:
            r=next(x for x in selected if x['experiment']==f'XPF16-Q{k:02d}' and x['method']==m)
            lines.append(f"| {float(r['target_coverage'])*100:.0f}% | {m} | {float(r['coverage'])*100:.3f}% | {float(r['width']):.5f} | {float(r['interval_score']):.5f} |")
    lines += ['', '完整 49×4×Raw/Online 结果见 `macro49.csv`；逐 Re、u/v、预测帧结果见 `test_group_lead.csv`，逐轨迹、逐窗口结果见分片 CSV。', '',
              '49 个独立头的 98 个点在概率轴上排列；50% 为 49%/51% 的插值。CRPS 采用相邻百分位的梯形积分，0%–1% 与 99%–100% 使用常数分位尾部假设；独立模型交叉时分别报告原值和单调重排结果。PIT 为插值后的分位函数反演，并在分布原子处采用中点。8/32/256 的经验样本 CRPS 与 XPF 分位积分 CRPS 是不同估计量。', '',
              '随机样本以各自的 90% 代理区间在线尺度变换后再计算经验分位；这不保证任意概率水平的覆盖。8/32 样本尤其无法稳定解析极端尾部。XPF 每组使用独立的双尾在线状态，只在完整窗口揭示后更新。', '',
              '图表提供 PNG 与 PDF，详见 `figures/`。逐窗口数据仅保存预测指标而非全部原始物理场。']
    p=OUT/'summary/REPORT.md';p.write_text('\n'.join(lines)+'\n')
    paths=sorted(Path(__file__).parent.glob('*.py'))
    (OUT/'summary/source_hashes.json').write_text(json.dumps({p.name:sha(p) for p in paths},indent=2)+'\n')
    print('REPORT_COMPLETE',flush=True)

if __name__=='__main__':main()
