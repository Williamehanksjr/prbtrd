import json
import math
import os
from collections import deque


class OnlineLearner:
    def __init__(self, state_path: str, horizon_seconds: int = 1200, learning_rate: float = 0.05):
        self.state_path = state_path
        self.horizon = horizon_seconds
        self.lr = learning_rate

        self.weights = {"bias": 0.0, "ret": 0.0, "vol_ratio": 0.0}
        self.pending = deque()
        self.correct = 0
        self.total = 0

        self._load()

    def _load(self):
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r") as f:
                    data = json.load(f)
                self.weights = data.get("weights", self.weights)
                self.correct = int(data.get("correct", 0))
                self.total = int(data.get("total", 0))
            except Exception:
                pass

    def _save(self):
        try:
            with open(self.state_path, "w") as f:
                json.dump(
                    {
                        "weights": self.weights,
                        "correct": self.correct,
                        "total": self.total,
                    },
                    f,
                )
        except Exception:
            pass

    def _sigmoid(self, x: float) -> float:
        x = max(-60.0, min(60.0, x))
        return 1.0 / (1.0 + math.exp(-x))

    def step(self, now_ts, prices, volumes):
        if len(prices) < 20 or len(volumes) < 20:
            return {}

        p0 = float(prices[-1])
        p5 = float(prices[-5]) if len(prices) >= 5 else p0
        vol_now = float(volumes[-1])
        vol_avg = sum(float(v) for v in volumes[-20:]) / 20.0 if len(volumes) >= 20 else max(vol_now, 1.0)

        ret = (p0 / p5 - 1.0) if p5 else 0.0
        vol_ratio = (vol_now / vol_avg) if vol_avg else 1.0

        x = {
            "bias": 1.0,
            "ret": float(ret),
            "vol_ratio": float(vol_ratio - 1.0),
        }

        z = sum(float(self.weights.get(k, 0.0)) * float(v) for k, v in x.items())
        p_up = self._sigmoid(z)

        self.pending.append((float(now_ts), float(p0), x, float(p_up)))

        while self.pending and (float(now_ts) - float(self.pending[0][0])) > float(self.horizon):
            ts_old, price_then, x_old, p_old = self.pending.popleft()
            y = 1.0 if float(p0) > float(price_then) else 0.0

            self.total += 1
            pred_up = p_old >= 0.5
            actual_up = y == 1.0
            if pred_up == actual_up:
                self.correct += 1

            error = y - p_old
            for k, v in x_old.items():
                self.weights[k] = float(self.weights.get(k, 0.0)) + self.lr * float(error) * float(v)

        self._save()

        return {
            "probability_up": float(p_up),
            "edge": float((p_up - 0.5) * 2.0),
            "accuracy": float(self.correct / self.total) if self.total > 0 else 0.0,
            "pending_count": int(len(self.pending)),
            "ready": bool(self.total > 20),
        }
