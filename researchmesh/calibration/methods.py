import math
from ..contracts import fail


# ---------------------------------------------------------------------------
# 校准方法一：Platt 缩放（逻辑回归 a*score+b，梯度下降拟合，纯标准库）
# ---------------------------------------------------------------------------
class PlattCalibrator:
    name = "platt_logistic"

    def __init__(self, a=0.0, b=0.0):
        self.a, self.b = a, b

    @classmethod
    def fit(cls, points, iterations=500, learning_rate=0.05):
        if not points:
            fail("input", "cannot fit calibrator without training points")
        a = b = 0.0
        # 分数缩放到 [-1,1]，改善梯度尺度
        xs = [(point["score"] - 50.0) / 50.0 for point in points]
        ys = [point["y"] for point in points]
        n = len(points)
        for _ in range(iterations):
            ga = gb = 0.0
            for x, y in zip(xs, ys):
                p = _sigmoid(a * x + b)
                error = p - y
                ga += error * x
                gb += error
            a -= learning_rate * ga / n
            b -= learning_rate * gb / n
        # 转回原始 0..100 分数域： sigmoid(a/50*score + (b-a))
        return cls(a / 50.0, b - a)

    def predict(self, score):
        return _sigmoid(self.a * score + self.b)

    def params(self):
        return {"a": _round6(self.a), "b": _round6(self.b)}


# ---------------------------------------------------------------------------
# 校准方法二：等渗回归（PAVA，单调经验概率，纯标准库）
# ---------------------------------------------------------------------------
class IsotonicCalibrator:
    name = "isotonic_pava"

    def __init__(self, blocks=None):
        # blocks: [(score_threshold, pooled_probability)] 按分数升序
        self.blocks = blocks or []

    @classmethod
    def fit(cls, points):
        if not points:
            fail("input", "cannot fit calibrator without training points")
        ordered = sorted(points, key=lambda point: (point["score"], -point["y"]))
        # 初始块：每个点一个块 (score, 概率和, 权重)
        blocks = [[point["score"], float(point["y"]), 1] for point in ordered]
        merged = True
        while merged:
            merged = False
            for i in range(len(blocks) - 1):
                if blocks[i][1] / blocks[i][2] > blocks[i + 1][1] / blocks[i + 1][2]:
                    left, right = blocks[i], blocks[i + 1]
                    # 同分块先合并；违反单调序则池化，阈值取权重均值
                    threshold = (
                        left[0] * left[2] + right[0] * right[2]
                    ) / (left[2] + right[2])
                    blocks[i : i + 2] = [
                        [threshold, left[1] + right[1], left[2] + right[2]]
                    ]
                    merged = True
                    break
        pooled = [
            (_round6(block[0]), _round6(block[1] / block[2]), block[2])
            for block in blocks
        ]
        return cls(pooled)

    def predict(self, score):
        if not self.blocks:
            fail("state", "isotonic calibrator is empty")
        # 找到第一个阈值 >= score 的块；score 超出范围时夹到端点概率
        for threshold, probability, _weight in self.blocks:
            if score <= threshold:
                return probability
        return self.blocks[-1][1]

    def params(self):
        return {
            "blocks": [
                {"score": threshold, "probability": probability, "weight": weight}
                for threshold, probability, weight in self.blocks
            ]
        }


class RawScoreCalibrator:
    """未校准基线：直接把规则分数除以 100 当概率，用于对照，不参与方法选择。"""

    name = "raw_score_baseline"

    def predict(self, score):
        return score / 100.0

    def params(self):
        return {"mapping": "score/100"}


METHODS = {
    PlattCalibrator.name: PlattCalibrator,
    IsotonicCalibrator.name: IsotonicCalibrator,
}


# ---------------------------------------------------------------------------
# 指标：可靠性分箱、Brier、精确率、召回率
# ---------------------------------------------------------------------------
def evaluate(points, calibrator, bins=10):
    if not isinstance(bins, int) or not 2 <= bins <= 20:
        fail("input", "bins must be between 2 and 20")
    if not points:
        fail("input", "evaluation requires at least one point")
    rows = []
    brier = 0.0
    for point in points:
        probability = _clamp(calibrator.predict(point["score"]))
        rows.append({**point, "probability": probability})
        brier += (probability - point["y"]) ** 2
    brier = brier / len(points)

    boundaries = [index / bins for index in range(bins + 1)]
    reliability = []
    ece = 0.0
    for index in range(bins):
        low, high = boundaries[index], boundaries[index + 1]
        members = [
            row
            for row in rows
            if (low <= row["probability"] < high)
            or (index == bins - 1 and row["probability"] == high)
        ]
        if not members:
            reliability.append(
                {
                    "range": [_round6(low), _round6(high)],
                    "count": 0,
                    "mean_predicted": None,
                    "observed_frequency": None,
                    "same": 0,
                    "different": 0,
                }
            )
            continue
        mean_predicted = sum(row["probability"] for row in members) / len(members)
        positives = sum(row["y"] for row in members)
        observed = positives / len(members)
        ece += len(members) / len(rows) * abs(mean_predicted - observed)
        reliability.append(
            {
                "range": [_round6(low), _round6(high)],
                "count": len(members),
                "mean_predicted": _round6(mean_predicted),
                "observed_frequency": _round6(observed),
                "same": positives,
                "different": len(members) - positives,
            }
        )

    return {
        "n": len(points),
        "brier": _round6(brier),
        "expected_calibration_error": _round6(ece),
        "reliability_bins": reliability,
        "rows": [
            {
                "pair_id": row["pair_id"],
                "score": row["score"],
                "probability": _round6(row["probability"]),
                "truth": "same" if row["y"] == 1 else "different",
            }
            for row in sorted(rows, key=lambda item: item["pair_id"])
        ],
    }


def precision_recall_at(points, calibrator, low, high):
    """三档策略下的精确率/召回率。

    概率 >= high 判建议同一，<= low 判建议不同，其余转人工。
    精确率针对“建议同一”里的真实同一比例；召回率针对真实同一被建议同一的比例。
    """
    tp = fp = fn = tn = manual = 0
    for point in points:
        probability = _clamp(calibrator.predict(point["score"]))
        if probability + 1e-12 >= high:
            if point["y"] == 1:
                tp += 1
            else:
                fp += 1
        elif probability - 1e-12 <= low:
            if point["y"] == 0:
                tn += 1
            else:
                fn += 1
        else:
            manual += 1
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    auto_rate = (tp + fp + tn + fn) / len(points)
    return {
        "low": _round6(low),
        "high": _round6(high),
        "confusion": {
            "suggest_same_correct": tp,
            "suggest_same_wrong": fp,
            "suggest_different_correct": tn,
            "suggest_different_wrong": fn,
            "manual": manual,
        },
        "precision": _round6(precision) if precision is not None else None,
        "recall": _round6(recall) if recall is not None else None,
        "automation_rate": _round6(auto_rate),
    }


def wilson_upper(successes, trials, z=1.959964):
    """二项比例的 Wilson 95% 置信区间上界，作为误合并风险上限。"""
    if trials == 0:
        return None
    phat = successes / trials
    denom = 1 + z * z / trials
    center = (phat + z * z / (2 * trials)) / denom
    half = (
        z * math.sqrt(phat * (1 - phat) / trials + z * z / (4 * trials * trials))
    ) / denom
    return _clamp(center + half)


def choose_thresholds(points, calibrator, risk_cap=0.05, grid=20, min_auto=1):
    """按误合并风险上限选择“建议同一 / 转人工 / 建议不同”双阈值。

    high 取满足“建议同一集合的 Wilson 风险上界 <= risk_cap”的最小概率门槛；
    low 取满足“建议不同集合中漏同一的 Wilson 风险上界 <= risk_cap”的最大
    概率门槛。样本不足或全程超限则返回 None，调用方必须降级为全人工。
    """
    if not isinstance(grid, int) or not 2 <= grid <= 100:
        fail("input", "grid must be between 2 and 100")
    if not 0.0 < risk_cap < 1.0:
        fail("input", "risk_cap must be strictly between 0 and 1")
    # 网格覆盖闭区间端点：完美分离时允许在接近 0/1 的极值处取阈值
    levels = [index / grid for index in range(grid + 1)]
    high = None
    high_report = None
    for candidate in reversed(levels[1:]):
        group = [
            point
            for point in points
            if _clamp(calibrator.predict(point["score"])) + 1e-12 >= candidate
        ]
        if len(group) < min_auto:
            continue
        wrong = sum(1 for point in group if point["y"] == 0)
        bound = wilson_upper(wrong, len(group))
        if bound is not None and bound <= risk_cap:
            high = candidate
            high_report = {
                "threshold": candidate,
                "auto_same": len(group),
                "wrong_merges": wrong,
                "wilson_risk_upper": _round6(bound),
            }
            break
    # 低阈值只在高阈值下方寻找，保证中间始终留有人工带
    low = None
    low_report = None
    if high is not None:
        for candidate in levels[:-1]:
            if candidate >= high:
                break
            group = [
                point
                for point in points
                if _clamp(calibrator.predict(point["score"])) - 1e-12 <= candidate
            ]
            if len(group) < min_auto:
                continue
            missed_same = sum(1 for point in group if point["y"] == 1)
            bound = wilson_upper(missed_same, len(group))
            if bound is not None and bound <= risk_cap:
                low = candidate
                low_report = {
                    "threshold": candidate,
                    "auto_different": len(group),
                    "missed_same": missed_same,
                    "wilson_risk_upper": _round6(bound),
                }
            else:
                break
    return {
        "low": _round6(low) if low is not None else None,
        "high": _round6(high) if high is not None else None,
        "risk_cap": risk_cap,
        "high_search": high_report,
        "low_search": low_report,
        "usable": low is not None and high is not None,
    }


def classify(probability, low, high):
    """按双阈值给出三档建议，比较带容差，与阈值搜索保持一致。"""
    probability = _clamp(probability)
    if probability + 1e-12 >= high:
        return "suggest_same"
    if probability - 1e-12 <= low:
        return "suggest_different"
    return "manual_review"


def _sigmoid(value):
    if value >= 0:
        ez = math.exp(-value)
        return 1.0 / (1.0 + ez)
    ez = math.exp(value)
    return ez / (1.0 + ez)


def _clamp(value):
    # 余量与阈值比较容差一致，保证完美预测仍能命中端点阈值 0/1
    return max(1e-12, min(1.0 - 1e-12, float(value)))


def _round6(value):
    return round(float(value), 6)
