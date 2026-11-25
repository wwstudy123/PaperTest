# 本地实测终点表（不同预算/目标，不作方法排名）

| 版本 | 优化目标 | 初始TPS | 最终TPS | 初始P95(ms) | 最终P95(ms) | 有效测量 | 实际API/tokens |
|---|---|---:|---:|---:|---:|---:|---:|
| ADMTune TPS prototype | TPS maximize | 2449.19 | 3807.49 | 18.28 | 68.05 | 71 | 13/23187 |
| Joint pilot | TPS/P95 maximize | 2449.99 | 2396.99 | 18.28 | 18.61 | 7 | 5/10641 |
| Legacy GP | TPS/P95 maximize | 2439.74 | 2381.40 | 18.28 | 18.61 | 7 | 4/8877 |
| Repeat GP (recovered) | TPS/P95 maximize | 2432.76 | 2487.79 | 18.28 | 17.63 | 7 | 4/9116 |

恢复臂最后仍是初始配置；终点变动不能当作调优收益。文献数值单独存放在published_reference_endpoints.csv，不能拿本地Default给文献方法计算统一提升率。
