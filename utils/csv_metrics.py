import csv
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Number
from pathlib import Path
from typing import Any


_KEY_SAFE_RE = re.compile(r"[^0-9a-zA-Z_\\.]+")


def _sanitize_key(key: str) -> str:
    key = str(key)
    key = _KEY_SAFE_RE.sub("_", key).strip("_")
    return key or "key"


def _is_scalar(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (str, bool, int, float)):
        return True
    if isinstance(value, Number):
        return True
    return False


def _to_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, Path):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)


@dataclass
class CsvMetricsSink:
    """
    以 CSV 的形式记录每个 epoch 的指标。

    - 自动扩展列：当出现新指标 key 时，重写整表并补齐缺失值；
    - 自动去重：相同 epoch 再写入会覆盖旧行；
    - 自动扁平化：对 list/tuple（小且为标量）做展开，对 dict 提取一层标量字段。
    """

    path: Path
    max_list_len: int = 64

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self._records: dict[int, dict[str, str]] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.path.is_file():
            return
        try:
            with self.path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if not row:
                        continue
                    epoch_raw = row.get("epoch")
                    if epoch_raw is None or epoch_raw == "":
                        continue
                    try:
                        epoch = int(float(epoch_raw))
                    except ValueError:
                        continue
                    self._records[epoch] = {k: (v if v is not None else "") for k, v in row.items()}
        except Exception:
            # 读取失败时不阻塞训练：按“无历史”继续。
            self._records = {}

    def update(self, values: Mapping[str, Any]) -> None:
        flattened = self._flatten(values)
        if "epoch" not in flattened:
            raise KeyError("CSV 指标记录缺少必需字段：epoch")
        try:
            epoch = int(float(flattened["epoch"]))
        except ValueError as exc:
            raise ValueError(f"epoch 无法解析为整数：{flattened['epoch']}") from exc
        flattened["epoch"] = epoch
        row = {k: _to_cell(v) for k, v in flattened.items()}
        self._records[epoch] = row
        self._write_all()

    def _flatten(self, values: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for raw_key, raw_value in values.items():
            key = _sanitize_key(raw_key)
            if raw_value is None:
                out[key] = None
                continue
            if _is_scalar(raw_value):
                out[key] = raw_value
                continue

            if isinstance(raw_value, (list, tuple)):
                if len(raw_value) <= self.max_list_len and all(_is_scalar(v) for v in raw_value):
                    for idx, item in enumerate(raw_value):
                        out[f"{key}_{idx}"] = item
                else:
                    # 避免把超长曲线/大对象塞进 CSV。
                    continue
                continue

            if isinstance(raw_value, Mapping):
                # 仅提取一层标量字段（比如 results_json 里的 precision/recall/f1/score_threshold）。
                for sub_key, sub_val in raw_value.items():
                    if _is_scalar(sub_val):
                        out[f"{key}.{_sanitize_key(sub_key)}"] = sub_val
                continue

            # 其他类型：尽量 stringify，但避免超大对象；这里保守跳过。
        return out

    def _fieldnames(self) -> list[str]:
        keys: set[str] = set()
        for row in self._records.values():
            keys.update(row.keys())
        keys.discard("")
        if not keys:
            return ["epoch"]
        fixed = []
        for k in ("epoch", "epoch_time"):
            if k in keys:
                fixed.append(k)
                keys.discard(k)
        return fixed + sorted(keys)

    def _write_all(self) -> None:
        fieldnames = self._fieldnames()
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        with tmp_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for epoch in sorted(self._records.keys()):
                row = self._records[epoch]
                writer.writerow({name: row.get(name, "") for name in fieldnames})
        tmp_path.replace(self.path)
