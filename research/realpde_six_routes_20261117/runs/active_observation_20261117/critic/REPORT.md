# 直接学习观测收益

完整相同同化器比较见[总报告](../REPORT.md)。

本模型在开发集胜过风险选点，但在late反向、future区间跨零；没有稳定转移证据。协议见源码CRITIC_PROTOCOL.md。

split,budget,method,reference,gain_percent,ci_low,ci_high,wins,n
calibration,8,critic,risk,1.5295886132687952,-0.19588296266652316,2.582594308666951,7,10
development,8,critic,risk,0.7000334031380229,0.009016124741040868,0.9570540041664657,8,12
late,8,critic,risk,-0.9737753261868987,-1.8280186363120814,-0.051669529925986724,4,12
future,8,critic,risk,-0.24536414121816286,-1.0392748473989277,0.42282863085392164,4,12
