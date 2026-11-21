# 冻结条件适配核查

协议、固定清单、权重哈希和结果在 `results/conditioning_failure_audit/`。只检查有效 C_aug/seed42/step0、250、1000；训练子集12条、开发12条、k5/start400/repeat0。运行不训练，也不访问新的测试集评分。

在仓库根目录使用已有数据、原生缓存和有效checkpoint，向新目录复跑：

```bash
source ../../env.sh
export DIPRED_FAILURE_OUT="$PWD/results/conditioning_failure_audit_reproduction"
mkdir -p "$DIPRED_FAILURE_OUT"
python research/conditioning_failure_audit/run.py > "$DIPRED_FAILURE_OUT/run.log" 2>&1
python research/conditioning_failure_audit/summarize.py > "$DIPRED_FAILURE_OUT/summarize.log" 2>&1
python research/conditioning_failure_audit/verify.py > "$DIPRED_FAILURE_OUT/verify.log" 2>&1
```

输入清单和不可变协议从已归档的 `results/conditioning_failure_audit/` 读取；复跑输出写上述新目录。程序拒绝覆盖已启动结果。需要原HDF5、旧缓存和checkpoint，路径/hash见manifest/config；资源或数值核验失败时记录STOP并停止，不自动重抽样。step0复用原始S5；step250/1000各生成一次完整20帧并同时评价三个时段，预测及delta记录之后才读取真值。`score_summary.csv`和`delta_summary.csv`分别为两张核心汇总；逐轨迹结果及校验、完整解释见结果目录。
