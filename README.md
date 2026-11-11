# 研合 科研成果融合

融合多个来源的成果记录，维护归一化词典、候选匹配、身份决策和不可变发布。记录键为 source:id；来源完整快照撤回未出现的记录，部分更新保留它们。

## 规则分数校准实验

候选证据的规则分数（0–100）是启发式打分，不是概率，不能直接按“80 分=八成”设自动通过线。`label` 与 `calibration` 动作组提供完整的匹配校准实验，全部只依赖标准库与 SQLite：

1. **人工标签台账**（`label.record`）：只接受人工确认的 `same`、`different`、`unknown` 标签并记录来源、备注与时间；既有合并决定不会被当作真值。同对矛盾标签在 `label.conflicts` 中可见，冻结前必须撤回或和解；`label.withdraw` 保留撤回历史而非删除。
2. **冻结与切分**（`calibration.freeze_dataset`）：快照每条边当时的规则分数、特征证据和 data/dictionary/graph 修订号。按“共享记录即连通”的连通分量整体切分 train/validation/holdout，同一记录及其候选边不会漏到两个桶；切分确定、无随机。分量超过 `max_component_fraction` 时默认阻塞（`blocked`），显式选择 `large_component=degrade` 才降级放入训练集。存在硬冲突或冻结后修订号移动都会被拒绝。
3. **训练与验证**（`calibration.run_validation`）：在训练集拟合两种校准方法——Platt 逻辑缩放与等渗回归（PAVA），并附未校准基线对照；验证集报告可靠性分箱、Brier、精确率、召回率，按验证 Brier 选主方法。阈值只在验证集上选择：高阈值使“建议同一”集合的误合并 Wilson 95% 风险上界不超过 `risk_cap`，低阈值对称约束“建议不同”的漏同一风险，中间始终保留人工带。
4. **留出集一次报告**（`calibration.evaluate_holdout`）：同一冻结数据集只能报告一次，重复调用被拒绝；阈值不允许用留出结果回调。样本不足、标签矛盾或风险上限不可证明时，`auto_threshold_usable=false`，不产生可自动使用的阈值。
5. **建议入队**（`calibration.suggest_batch`）：结果只作为 `suggest_same`、`manual_review`、`suggest_different` 建议进入现有审查队列，绝不修改身份图。批次按输入摘要幂等，重复提交同一批建议不会重复建审查项。

## 运行环境

使用 Python 3.11，核心功能依赖标准库和 SQLite。演示和测试只使用本地合成数据，不需要账户、设备或外部运行服务。请从项目根目录执行命令。

## 检查与运行

构建检查验证语法，测试使用临时数据目录，不会连接远端服务。

```sh
python3 -m compileall -q researchmesh
python3 -m unittest discover -s tests -v
python3 -m researchmesh demo
```

演示会展示产品的实际数据处理和输出，运行结束自行退出。查看现有接口：

```sh
python3 -m researchmesh --db :memory: describe
```

## 目录与数据

researchmesh/ 保存领域逻辑和请求入口，tests/ 包含正常行为、边界与恢复测试。例子中的标识和记录为合成资料，不包含真实人员或生产测量。业务目录和本地数据库由调用者指定，应与源码分开保存。

## 干净环境复现

容器定义固定运行时版本。构建时准备环境，运行时无需网络；每次启动会复制只读源码到临时工作目录，测试产生的文件不会写回仓库。

```sh
docker build -t local-product-check .
docker run --rm --network none local-product-check
```
