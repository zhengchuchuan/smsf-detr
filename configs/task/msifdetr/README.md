## 命名规范

本目录下任务配置文件统一使用如下命名格式（全小写 + 下划线）：

`数据集_模型_检测类型_融合方式_注释.yaml`

字段含义建议如下：

- `数据集`：`coco_rgb` / `coco_rgb_msi` / `oil_rgb` / `oil_rgb_msi` 等（与 `data.dataset_file` / 数据来源一致）
- `模型`：`msifdetr_small` / `msifdetr_base` 等（与 `model` 规模/变体一致）
- `检测类型`：`det`（检测）/ `seg`（分割）
- `融合方式`：
  - `no_fuse`：不融合（RGB-only 或双流不融合输出）
  - `stack`：RGB+MSI 通道堆叠输入
  - `dual`：双流（具体融合策略写在注释里，如 `dual_sum` / `dual_concat`）
- `注释`：用于记录关键实验差异（如 `match_best_pretrain_train`、`ablate_norm` 等）

示例：

- `coco_rgb-msifdetr_small-det-no_fuse-match_best_pretrain_train.yaml`
