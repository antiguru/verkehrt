# Top-k kernels: bench

The fused kernels `topk_<t>` and `topk2_<a>_<b>` compute `slice(argsort(...), 0, e)` without sorting every row when `e` is at most `TOPK_MAX`.
This record compares them with the unfused plan, `argsort` followed by `slice`, on uniform and descending keys at 80,000 and 1M rows.
Descending keys are the worst case of the selection, because every row passes the threshold and enters the buffer.
The numbers fix `TOPK_MAX` at 64 and show the selection ahead of the full sort on descending input up to `e` = 128.
They also show why the kernel buffers rows unsorted and trims them with a partial selection.

## Method

The groups `tier2/topk-<shape>-e<e>-<size>` and `tier2/topk2-<shape>-e<e>-<size>` of `benches/tier1.rs` run each program on two interpreters with default options, one that fuses (`topk`) and one with `no_fusion` (`argsort-slice`).
The one-column program is `slice(argsort(x), 0, e)` and the two-column program is `slice(argsort(x, y), 0, e)`, over `i64` columns, and every variant's answer is checked against a sorted reference before it is timed.
The uniform shape draws the first column from 1,000 distinct values, so keys tie, and the second column over the full range.
The descending shape has a first column `(n - i) / 4` and a second column `-i`, so the composite keys are strictly descending.
Each figure is the middle estimate of the criterion time interval, with 10 samples, 0.5 s of warm-up and 3 s of measurement per variant, in microseconds.
The cut sizes come from `VERKEHRT_TOPK_E`, and the default is 10 and 64.
The host has CPU boost off and a noise band of about 10%, and the one-minute load average was between 2.9 and 3.6 when each run started.

```
# final code, e = 10 and 64
VERKEHRT_TOPK_E unset  cargo bench --bench tier1 -- 'tier2/topk'
# the sweeps, e = 10, 64, 128, 256 (the sorted-buffer sweep also used 32)
VERKEHRT_TOPK_E=10,64,128,256  cargo bench --bench tier1 -- 'tier2/topk'
```

## Result

On the final code the selection wins every measured cell.
The smallest speedup is 1.7x, one column at 80,000 descending rows with `e` = 10, and the largest is 28.6x, two columns at 1M uniform rows.
The uniform shape gains most, at 6.6x to 28.6x against 1.7x to 9.97x for descending keys.
The two-column kernels gain more than the one-column kernels, and the unfused two-column sort takes 29 ms at 1M uniform rows against 5 ms for one column.
A sweep of the final algorithm to `e` = 256 stays ahead of the full sort on descending input up to `e` = 128, with a lowest speedup of 1.28x for one column at 80,000 rows, and ties it at `e` = 256 (0.98x).
`TOPK_MAX` is 64, inside that range.

Final code, time of the fused kernel against the unfused plan:

```
tier2/topk-descending-e10-1M                       topk     3228.5us  argsort    11399.0us  x 3.53
tier2/topk-descending-e10-80k                      topk      260.3us  argsort      439.4us  x 1.69
tier2/topk-descending-e64-1M                       topk     2967.3us  argsort    11450.0us  x 3.86
tier2/topk-descending-e64-80k                      topk      237.5us  argsort      440.3us  x 1.85
tier2/topk-uniform-e10-1M                          topk      629.4us  argsort     5195.4us  x 8.25
tier2/topk-uniform-e10-80k                         topk       57.9us  argsort      416.8us  x 7.20
tier2/topk-uniform-e64-1M                          topk      689.0us  argsort     5203.3us  x 7.55
tier2/topk-uniform-e64-80k                         topk       63.3us  argsort      416.5us  x 6.58
tier2/topk2-descending-e10-1M                      topk     5139.2us  argsort    49058.0us  x 9.55
tier2/topk2-descending-e10-80k                     topk      405.8us  argsort     1731.1us  x 4.27
tier2/topk2-descending-e64-1M                      topk     4927.3us  argsort    49126.0us  x 9.97
tier2/topk2-descending-e64-80k                     topk      401.6us  argsort     1730.6us  x 4.31
tier2/topk2-uniform-e10-1M                         topk     1022.1us  argsort    29237.0us  x28.60
tier2/topk2-uniform-e10-80k                        topk       85.9us  argsort     2329.1us  x27.11
tier2/topk2-uniform-e64-1M                         topk     1033.2us  argsort    29285.0us  x28.34
tier2/topk2-uniform-e64-80k                        topk       92.1us  argsort     2317.5us  x25.17
```

## Rejected variants

A buffer kept sorted, with an insertion per row that beats the threshold, lost on descending input at every size, at 0.25x to 0.55x for one column with `e` = 10.
An unsorted buffer of `2e` rows, trimmed when full, fixed the large inputs but lost at 80,000 descending rows with `e` = 10 (0.67x), because a trim of 20 rows runs every 10 rows.
Letting the buffer hold at least 256 rows beyond the kept ones, `BATCH`, amortizes the trim and gives the final numbers.

Sorted buffer, sweep to `e` = 256:

```
tier2/topk-descending-e10-1M                       topk    20647.0us  argsort    11320.0us  x 0.55
tier2/topk-descending-e10-80k                      topk     1656.5us  argsort      420.7us  x 0.25
tier2/topk-descending-e128-1M                      topk    40729.0us  argsort    11343.0us  x 0.28
tier2/topk-descending-e128-80k                     topk     3267.6us  argsort      420.3us  x 0.13
tier2/topk-descending-e256-1M                      topk    55979.0us  argsort    11342.0us  x 0.20
tier2/topk-descending-e256-80k                     topk     4328.1us  argsort      419.5us  x 0.10
tier2/topk-descending-e32-1M                       topk    27027.0us  argsort    11321.0us  x 0.42
tier2/topk-descending-e32-80k                      topk     2186.5us  argsort      417.0us  x 0.19
tier2/topk-descending-e64-1M                       topk    34173.0us  argsort    11363.0us  x 0.33
tier2/topk-descending-e64-80k                      topk     2735.9us  argsort      419.7us  x 0.15
tier2/topk-uniform-e10-1M                          topk      883.5us  argsort     5148.7us  x 5.83
tier2/topk-uniform-e10-80k                         topk       76.5us  argsort      400.1us  x 5.23
tier2/topk-uniform-e128-1M                         topk      954.3us  argsort     5155.2us  x 5.40
tier2/topk-uniform-e128-80k                        topk      106.5us  argsort      400.2us  x 3.76
tier2/topk-uniform-e256-1M                         topk      992.1us  argsort     5168.6us  x 5.21
tier2/topk-uniform-e256-80k                        topk      148.1us  argsort      399.2us  x 2.70
tier2/topk-uniform-e32-1M                          topk      923.2us  argsort     5155.9us  x 5.58
tier2/topk-uniform-e32-80k                         topk       77.2us  argsort      399.5us  x 5.18
tier2/topk-uniform-e64-1M                          topk      903.2us  argsort     5160.6us  x 5.71
tier2/topk-uniform-e64-80k                         topk       86.1us  argsort      398.8us  x 4.63
tier2/topk2-descending-e10-1M                      topk    25121.0us  argsort    48858.0us  x 1.94
tier2/topk2-descending-e10-80k                     topk     1985.9us  argsort     1718.0us  x 0.87
tier2/topk2-descending-e128-1M                     topk    54608.0us  argsort    48803.0us  x 0.89
tier2/topk2-descending-e128-80k                    topk     4361.1us  argsort     1716.7us  x 0.39
tier2/topk2-descending-e256-1M                     topk    76685.0us  argsort    48883.0us  x 0.64
tier2/topk2-descending-e256-80k                    topk     6122.4us  argsort     1718.0us  x 0.28
tier2/topk2-descending-e32-1M                      topk    35084.0us  argsort    48984.0us  x 1.40
tier2/topk2-descending-e32-80k                     topk     2823.7us  argsort     1719.7us  x 0.61
tier2/topk2-descending-e64-1M                      topk    43546.0us  argsort    49040.0us  x 1.13
tier2/topk2-descending-e64-80k                     topk     3481.6us  argsort     1717.8us  x 0.49
tier2/topk2-uniform-e10-1M                         topk     1362.5us  argsort    29163.0us  x21.40
tier2/topk2-uniform-e10-80k                        topk      112.4us  argsort     2204.9us  x19.62
tier2/topk2-uniform-e128-1M                        topk     1428.1us  argsort    29150.0us  x20.41
tier2/topk2-uniform-e128-80k                       topk      153.5us  argsort     2207.3us  x14.38
tier2/topk2-uniform-e256-1M                        topk     1513.0us  argsort    29144.0us  x19.26
tier2/topk2-uniform-e256-80k                       topk      209.2us  argsort     2205.9us  x10.54
tier2/topk2-uniform-e32-1M                         topk     1368.3us  argsort    29183.0us  x21.33
tier2/topk2-uniform-e32-80k                        topk      120.0us  argsort     2206.7us  x18.38
tier2/topk2-uniform-e64-1M                         topk     1388.8us  argsort    29420.0us  x21.18
tier2/topk2-uniform-e64-80k                        topk      131.2us  argsort     2201.3us  x16.77
```

Unsorted buffer of `2e` rows, `e` = 10, 64, 128, 256:

```
tier2/topk-descending-e10-1M                       topk     7734.9us  argsort    11296.0us  x 1.46
tier2/topk-descending-e10-80k                      topk      616.1us  argsort      414.2us  x 0.67
tier2/topk-descending-e128-1M                      topk     5553.7us  argsort    11303.0us  x 2.04
tier2/topk-descending-e128-80k                     topk      443.7us  argsort      414.0us  x 0.93
tier2/topk-descending-e256-1M                      topk     5285.7us  argsort    11236.0us  x 2.13
tier2/topk-descending-e256-80k                     topk      427.0us  argsort      413.3us  x 0.97
tier2/topk-descending-e64-1M                       topk     5830.7us  argsort    11489.0us  x 1.97
tier2/topk-descending-e64-80k                      topk      465.0us  argsort      414.2us  x 0.89
tier2/topk-uniform-e10-1M                          topk      485.6us  argsort     5106.1us  x10.51
tier2/topk-uniform-e10-80k                         topk       53.2us  argsort      390.3us  x 7.33
tier2/topk-uniform-e128-1M                         topk      636.6us  argsort     5063.4us  x 7.95
tier2/topk-uniform-e128-80k                        topk       66.9us  argsort      390.5us  x 5.84
tier2/topk-uniform-e256-1M                         topk      689.2us  argsort     5061.8us  x 7.34
tier2/topk-uniform-e256-80k                        topk       73.9us  argsort      391.9us  x 5.30
tier2/topk-uniform-e64-1M                          topk      428.2us  argsort     5057.0us  x11.81
tier2/topk-uniform-e64-80k                         topk       61.4us  argsort      390.1us  x 6.35
tier2/topk2-descending-e10-1M                      topk    11892.0us  argsort    48663.0us  x 4.09
tier2/topk2-descending-e10-80k                     topk      930.0us  argsort     1725.8us  x 1.86
tier2/topk2-descending-e128-1M                     topk     8741.9us  argsort    48607.0us  x 5.56
tier2/topk2-descending-e128-80k                    topk      703.8us  argsort     1725.3us  x 2.45
tier2/topk2-descending-e256-1M                     topk     8471.9us  argsort    48748.0us  x 5.75
tier2/topk2-descending-e256-80k                    topk      725.3us  argsort     1724.8us  x 2.38
tier2/topk2-descending-e64-1M                      topk     9633.0us  argsort    48602.0us  x 5.05
tier2/topk2-descending-e64-80k                     topk      779.4us  argsort     1724.7us  x 2.21
tier2/topk2-uniform-e10-1M                         topk      824.2us  argsort    30334.0us  x36.80
tier2/topk2-uniform-e10-80k                        topk       68.5us  argsort     2265.0us  x33.05
tier2/topk2-uniform-e128-1M                        topk      845.5us  argsort    30197.0us  x35.71
tier2/topk2-uniform-e128-80k                       topk       83.6us  argsort     2267.8us  x27.11
tier2/topk2-uniform-e256-1M                        topk      869.1us  argsort    30192.0us  x34.74
tier2/topk2-uniform-e256-80k                       topk       95.8us  argsort     2269.3us  x23.69
tier2/topk2-uniform-e64-1M                         topk      833.3us  argsort    30267.0us  x36.32
tier2/topk2-uniform-e64-80k                        topk       75.4us  argsort     2261.6us  x30.01
```

Final algorithm, `e` = 10, 64, 128, 256:

```
tier2/topk-descending-e10-1M                       topk     3182.5us  argsort    11392.0us  x 3.58
tier2/topk-descending-e10-80k                      topk      258.7us  argsort      416.8us  x 1.61
tier2/topk-descending-e128-1M                      topk     4047.7us  argsort    11362.0us  x 2.81
tier2/topk-descending-e128-80k                     topk      324.9us  argsort      416.0us  x 1.28
tier2/topk-descending-e256-1M                      topk     5196.9us  argsort    11395.0us  x 2.19
tier2/topk-descending-e256-80k                     topk      422.4us  argsort      415.9us  x 0.98
tier2/topk-descending-e64-1M                       topk     2923.1us  argsort    11327.0us  x 3.87
tier2/topk-descending-e64-80k                      topk      235.2us  argsort      415.1us  x 1.76
tier2/topk-uniform-e10-1M                          topk      372.0us  argsort     5011.9us  x13.47
tier2/topk-uniform-e10-80k                         topk       33.3us  argsort      401.9us  x12.08
tier2/topk-uniform-e128-1M                         topk      418.6us  argsort     5035.6us  x12.03
tier2/topk-uniform-e128-80k                        topk       43.2us  argsort      400.5us  x 9.27
tier2/topk-uniform-e256-1M                         topk      403.1us  argsort     5062.0us  x12.56
tier2/topk-uniform-e256-80k                        topk       52.1us  argsort      401.1us  x 7.70
tier2/topk-uniform-e64-1M                          topk      388.0us  argsort     5049.9us  x13.01
tier2/topk-uniform-e64-80k                         topk       39.4us  argsort      401.6us  x10.19
tier2/topk2-descending-e10-1M                      topk     4906.5us  argsort    50007.0us  x10.19
tier2/topk2-descending-e10-80k                     topk      391.9us  argsort     1727.8us  x 4.41
tier2/topk2-descending-e128-1M                     topk     8297.2us  argsort    50085.0us  x 6.04
tier2/topk2-descending-e128-80k                    topk      647.1us  argsort     1725.3us  x 2.67
tier2/topk2-descending-e256-1M                     topk     8263.9us  argsort    50317.0us  x 6.09
tier2/topk2-descending-e256-80k                    topk      666.4us  argsort     1727.2us  x 2.59
tier2/topk2-descending-e64-1M                      topk     4941.0us  argsort    50142.0us  x10.15
tier2/topk2-descending-e64-80k                     topk      395.0us  argsort     1728.7us  x 4.38
tier2/topk2-uniform-e10-1M                         topk      826.3us  argsort    29297.0us  x35.45
tier2/topk2-uniform-e10-80k                        topk       70.7us  argsort     2224.5us  x31.46
tier2/topk2-uniform-e128-1M                        topk      843.6us  argsort    29371.0us  x34.82
tier2/topk2-uniform-e128-80k                       topk       83.2us  argsort     2227.2us  x26.78
tier2/topk2-uniform-e256-1M                        topk      865.0us  argsort    29403.0us  x33.99
tier2/topk2-uniform-e256-80k                       topk       96.3us  argsort     2224.8us  x23.10
tier2/topk2-uniform-e64-1M                         topk      833.5us  argsort    29244.0us  x35.09
tier2/topk2-uniform-e64-80k                        topk       77.1us  argsort     2223.7us  x28.85
```
