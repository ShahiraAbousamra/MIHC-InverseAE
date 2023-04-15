from enum import Enum;

class CostFuncTypes(Enum):
    CROSS_ENTROPY = 1


class CNNLossFuncHelper:

    @staticmethod
    def cost_cross_entropy(logits, labels, class_weights, n_classes):
        pass;

    @staticmethod
    def cost_mse(logits, labels, class_weights, n_classes):
        pass;

