"""Independent two-sample PPI with a configurable interval function."""
import numpy as np
from ..base import Method
from ..data import PPIData


class PPI(Method):
    """Binary mean PPI with a predictor trained independently of inference."""
    name = 'PPI'
    data_type = PPIData

    def _estimate(self, data, target):
        residuals = data.labels - data.labeled_predictions
        predictions = data.unlabeled_predictions
        point = float(predictions.mean() + residuals.mean())
        se = float(np.sqrt(predictions.var(ddof=1) / len(predictions)
                           + residuals.var(ddof=1) / len(residuals)))
        return point, se, {
            'values': {'residuals': residuals, 'unlabeled_predictions': predictions},
            'labeled_count': len(residuals),
            'unlabeled_count': len(predictions),
            'prediction_weight': 1.0,
            'weight_strategy': 'fixed',
            'variance_design': 'independent_two_sample',
        }
