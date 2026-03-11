import math
from typing import Any, Dict, Optional


def _safe_index(seq, idx):
    try:
        return seq[idx]
    except Exception:
        return None


class WandbMetricLogger:
    def __init__(self, run) -> None:
        self.run = run

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return None if math.isnan(numeric) else numeric

    def _prepare_extended_metrics(self, results_json: Dict[str, Any], metric_prefix: str) -> Dict[str, float]:
        if not isinstance(results_json, dict):
            return {}

        metrics = {}

        def _get_first_present(*keys: str) -> Any:
            for key in keys:
                if key in results_json:
                    return results_json.get(key)
            return None

        def _maybe_add(name: str, value: Any):
            numeric = self._safe_float(value)
            if numeric is not None:
                metrics[f"{metric_prefix}/{name}"] = numeric

        _maybe_add("Precision", results_json.get("precision"))
        _maybe_add("Recall", results_json.get("recall"))
        _maybe_add("F1", results_json.get("f1"))
        _maybe_add("mAP50", results_json.get("map"))
        _maybe_add("mAP75", _get_first_present("map@75", "map75"))
        _maybe_add("mAP50_95", _get_first_present("map@50:95", "map50_95"))
        _maybe_add("ScoreThreshold", results_json.get("score_threshold"))

        return metrics

    def update(self, values: Dict[str, Any]) -> None:
        if not self.run or not isinstance(values, dict):
            return

        log_dict: Dict[str, Any] = {}
        epoch = values.get('epoch')
        if epoch is not None:
            log_dict['epoch'] = epoch

        if 'train_loss' in values:
            log_dict["Loss/Train"] = values['train_loss']
        if 'test_loss' in values:
            log_dict["Loss/Test"] = values['test_loss']

        if 'test_coco_eval_bbox' in values:
            coco_eval = values['test_coco_eval_bbox']
            ap50_90 = _safe_index(coco_eval, 0)
            ap50 = _safe_index(coco_eval, 1)
            ar50_90 = _safe_index(coco_eval, 8)
            if ap50_90 is not None:
                log_dict["Metrics/Base/AP50_90"] = ap50_90
            if ap50 is not None:
                log_dict["Metrics/Base/AP50"] = ap50
            if ar50_90 is not None:
                log_dict["Metrics/Base/AR50_90"] = ar50_90

        if 'ema_test_coco_eval_bbox' in values:
            ema_coco_eval = values['ema_test_coco_eval_bbox']
            ema_ap50_90 = _safe_index(ema_coco_eval, 0)
            ema_ap50 = _safe_index(ema_coco_eval, 1)
            ema_ar50_90 = _safe_index(ema_coco_eval, 8)
            if ema_ap50_90 is not None:
                log_dict["Metrics/EMA/AP50_90"] = ema_ap50_90
            if ema_ap50 is not None:
                log_dict["Metrics/EMA/AP50"] = ema_ap50
            if ema_ar50_90 is not None:
                log_dict["Metrics/EMA/AR50_90"] = ema_ar50_90

        if 'test_results_json' in values:
            log_dict.update(self._prepare_extended_metrics(values['test_results_json'], "Metrics/Base"))

        if 'ema_test_results_json' in values:
            log_dict.update(self._prepare_extended_metrics(values['ema_test_results_json'], "Metrics/EMA"))

        if log_dict:
            self.run.log(log_dict)

    def close(self) -> None:
        if not self.run:
            return
        try:
            self.run.finish()
        finally:
            self.run = None
