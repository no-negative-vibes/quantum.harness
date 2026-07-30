# Typed exterior category 第一轮结果

更新时间：2026-07-29

贡献人：籼至（GitHub `xianzhipan`，Codex 协助）

运行编号：`exterior-positive-category-v1-pilot`
run-spec digest：`329894802a07249a79de6b2fa7a7bcf694476e3582b95a5e2f6e0d5856df6f25`

## 先说人话

这轮确实搜索了我们自己的新候选，不是只写计划。

我们让矩阵带有“从状态 A 到状态 B”或“从 B 回到 A”的类型。只有首尾能接上的
矩阵顺序才是物理上允许的历史。目标是找到这样一种情况：

1. 所有允许的闭合历史都由一个精确证书保证非负，而且保证的是任意长度；
2. 如果把 A/B 标签抹掉、允许矩阵乱接，就能得到精确负权；
3. 这种正性又不能通过简单换坐标还原成已知 TN 机制。

第一版一共生成 `2,400` 个探索候选。结果是：

- `2,367` 个连第一道严格正性门都过不了；
- `32` 个虽然有正性证书，但到深度 5 都找不到“抹掉类型后为负”的词，尚未证明
  类型标签真的有用；
- `1` 个同时得到任意深度合法正性证书和精确负权 `-10`；
- 但最后这个对象能经状态依赖的 signed-permutation 换基化成已知 TN 正矩阵，
  所以不是新机制；
- 通过当前前门排重的新候选为 `0`，物理强候选也为 `0`。

这不是“什么都没做出来”。它完成了一个可复现的漏斗，找到并正确拦下了一个很会
伪装的假阳性，也明确关闭了 `support_aware_sparse-v1` 这套具体生成语法。但它没有
关闭整个 typed exterior 方向。

## 搜索对象

每个候选有两个对象 `a,b` 和三条边：

```text
forward-1 : a -> b
forward-2 : a -> b
return    : b -> a
```

时间顺序 `(e1,e2,...,eL)` 使用

```text
P = B_eL ... B_e2 B_e1.
```

对每个 exterior grade `k=0,...,n`，程序求对象依赖的精确有理数符号 chart。
若 chart 中每条边的 `Lambda^k(B_e)` 都逐项非负，则任意合法闭合历史都满足

```text
tr Lambda^k(P) >= 0
```

并由恒等式

```text
det(I+P) = sum_k tr Lambda^k(P)
```

得到任意长度的严格非负定理。深度上限 5 只用于寻找“抹掉类型后”的负反例，
不限制合法历史的正性证明。

## 协议

| 轴 | 取值 |
|---|---|
| 维数 | `4,5,6` |
| cell seeds | `14,77,2026072901,2026072902` |
| 每格候选 | `200` |
| 已知控制语法 | `coboundary_tn_control` |
| 探索语法 | `support_aware_sparse` |
| type-erasure 最大深度 | `5` |
| 合法历史压力深度 | `2,4,8,16,32` |

探索语法用恒等/三循环置换、一个正对角缩放和一至三个整数 shear 组成稀疏矩阵。
另放入同样大小的已知 TN coboundary 控制组，验证机器既能认出真证书，也能把它标成
已知校准而不是新发现。

Float64 只作便宜的拒绝筛选。所有 theorem chart、负权和保存的 witness 都重新使用
SymPy 精确有理数计算。

## 完整运行结果

完整性门：

```text
planned = 24
success = 24
failed  = 0
missing = 0
```

### 分类漏斗

| 组别 | 候选 | 无 chart | 已知校准 | signed-permutation TN | 未显示类型必要 | 边界零权 | 前门暂存候选 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 已知控制 | 2,400 | 0 | 1,754 | 0 | 43 | 603 | 0 |
| 我们的探索 | 2,400 | 2,367 | 0 | 1 | 32 | 0 | 0 |
| **合计** | **4,800** | **2,367** | **1,754** | **1** | **75** | **603** | **0** |

控制组的 `603` 个“边界零权”主要是精确零而不是数值算不准。它们只用于校准，
不参与新颖性计数。

### 实际计算量

| 操作 | 次数 | 口径 |
|---|---:|---|
| float edge-grade compound screens | 86,400 | 每个候选的三条边、全部 `n+1` 个 exterior grades |
| determinant word checks | 257,669 | 合法/擦除类型后的 word 权重检查 |
| exact rational replays | 44,577 | 对浮点命中或边界项的精确重算 |
| exact negative witnesses | 1,755 | 其中 1,754 个来自故意放入的已知控制，探索组只有 1 个 |
| legal stress evaluations | 480 | 深度 `2,4,8,16,32`；执行次数，不声称每次都是不同 word |

`86,400` 不能与旧的 `4,044,000` 个 determinant 主权重直接相加，因为前者数的是
exterior edge-grade compound matrices，不是同一种操作。

### 分维度探索结果

| 维数 | 候选 | 无 chart | 未显示类型必要 | 已知 TN 命中 | 前门暂存候选 |
|---|---:|---:|---:|---:|---:|
| 4 | 800 | 787 | 12 | 1 | 0 |
| 5 | 800 | 791 | 9 | 0 | 0 |
| 6 | 800 | 789 | 11 | 0 | 0 |

signed-permutation 穷举上限为 1,000 个对象置换。对两对象四维空间，
`(4!)^2=576`，所以四维排查完整；五、六维分别需要 14,400 和 518,400 个置换，
上限不足。不过本轮五、六维没有任何对象走到“有负 witness、需要做该排重”的门，
因此没有把 cap miss 误报为新候选。

## 唯一探索命中为什么不是新类

唯一命中位于 `cell-0013`、candidate index `155`：

```text
candidate seed = 15103403883834912679
candidate id   = 15526eb4720007ec3271d9c268589f86aef2143b351cf8b36b65a5fc23acda80
```

三条矩阵为

```text
forward-1 =
[[0, 1, 0, 0],
 [0, 0, 0, 1],
 [0, 2, 1, 0],
 [1, 0, 0, 0]]

forward-2 =
[[ 0, 1, 0, 0],
 [-2, 0, 0, 1],
 [ 0, 0, 2, 0],
 [ 1, 0, 0, 0]]

return =
[[-1, 0, 0, 1],
 [ 1, 0, 0, 0],
 [ 0, 0, 1, 0],
 [ 0, 2, 0, 0]]
```

一个合法闭路是

```text
(forward-1, return)
```

其各 exterior grade trace 精确为

```text
(1,5,9,7,2)
```

总权重为 `24`。grade chart 证明的不是这一条，而是所有合法闭路任意深度非负。

抹掉类型后，下面这个深度 5 的非法词给出

```text
(forward-1, forward-1, forward-2, forward-1, forward-2)
det(I+P) = -10.
```

它一度是合格候选，因为这套 grade charts 不能由单纯的一粒子对角符号 chart 诱导。
但完整四维 signed-permutation 排查在第 350 个对象置换找到

```text
a -> (2,1,0,3)
b -> (2,0,3,1)
```

并连同相应符号规范把三条 typed edges 同时化成 TN 非负坐标矩阵。因此它的
“合法顺序正、乱接会负”是真事实，但原因仍是已知的状态依赖 TN coboundary。

## 能说与不能说

可以说：

- typed exterior 的 exact certificate、type-erasure 反例和已知类排重流水线已经运行；
- `support_aware_sparse-v1` 的 2,400 个候选没有留下前门新候选；
- 找到一个精确深度 5 假阳性并用完整四维穷举证明它属于已知 TN 换基；
- 合法闭路正性证书是任意深度的，不是有限深度抽样。

不能说：

- typed exterior 整个方向已经被排空；
- 已找到新的正性矩阵类或新的无符号物理模型；
- 五、六维 signed-permutation TN 排查已经穷尽；
- 任意 primitive edge 都是单个真实指数；
- 已经完成 HS、Hamiltonian 或 pairing/Pfaffian 映射。

唯一命中的负词是混合词，不是负 singleton，因此不能用负 singleton 定理直接否定
每条 primitive edge 的真实对数。但该候选已经在更早的已知 TN 门被降级，所以本轮
没有继续花成本做 real-log 和 Hamiltonian 审计。

## 下一轮怎么改

继续盲目增加同一语法的随机数意义不大。下一轮应改候选分布：

1. 在生成阶段直接满足一部分 grade-chart 约束，避免 98.6% 样本死在第一门；
2. 直接商掉 diagonal/signed-permutation TN coboundary，不再反复生成旧类伪装；
3. 扩大到真正 grade-dependent、非一粒子诱导的有理 charts，并加入 invariant block、
   common metric、split/Kramers 和标准 representation lift 排重；
4. 只有数学候选穿过这些门后，再要求每条 primitive edge 有真实对数，并反推
   HS/Hamiltonian；
5. 含真实 pairing 的 Pfaffian/Spin 搜索保持独立，不用 determinant 平方替代。

因此这轮给出的不是“矿脉已经不存在”，而是一个明确反馈：

> 稀疏置换 + 对角缩放 + 少量 shear 的无条件随机生成，几乎总与 exterior
> sign constraints 冲突；偶尔命中又容易落回状态依赖 TN。下一轮必须条件化生成，
> 并在生成器层面先除掉已知 coboundary。

## 复现

协议和命令见
[`protocols/exterior-positive-category-v1/README.md`](../protocols/exterior-positive-category-v1/README.md)。

核心实现：

- `oracle/compound_cones.py`
- `oracle/exterior_category_search.py`
- `oracle/known_class_filters.py`
- `oracle/exterior_category_pilot.py`

运行代码 SHA256：

```text
exterior_category_pilot.py  4f1fe561f9afd79a5f3b986c88ead9874f4649db6090a9ea07c84c3830c496b6
exterior_category_search.py 0fa6d78d5f221e3142d5b275de95cf03f00e6b7ea263f6f129db6be233d8644d
known_class_filters.py       fc32996282f6915a884d221cb1326ad400fb4b9f4ecef49511adb9f21661dacb
```

运行前后完整测试：

```text
408 passed
```

大体积逐候选 manifests 位于 Git 忽略的
`tracks/qmc/results/no-negative-vibes/exterior-positive-category-v1-pilot/`；
专题汇总和唯一探索命中的小型机器可读证据保存在
`fixtures/exterior_positive_category_pilot.json`。
