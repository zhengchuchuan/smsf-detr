"""
阶段1等价基线：
- SMSFDETR 训练器先继承 RTMSFDETR 训练器；
- 保证训练/评估行为与 RTMSFDETR 一致，仅改变 trainer/model 标识。
"""

from engines.trainer.rtmsfdetr_trainer import RtmsfDetrTrainer


class SmsfDetrTrainer(RtmsfDetrTrainer):
    pass
