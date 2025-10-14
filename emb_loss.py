from pytorch_metric_learning import distances
from pytorch_metric_learning.losses import SupConLoss
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu


class EmbLoss():
    def __init__(self, name='kg', temperature=0.1):

        # Set distance metric
        distance = distances.CosineSimilarity()

        # Initialize loss function
        self.name = name
        if name == 'supcon':
            self.loss = SupConLoss(
                distance=distance,
                temperature=temperature
            )
        elif name == 'kg':
            # Adds distance calcuation between kg and self
            self.loss = SupConLossSelfDist(
                distance=distance,
                temperature=temperature
            )
        else:
            raise ValueError(f'Loss name {name} not recognized')

    def __call__(self, projections, y_label, y_kg_emb):
        if self.name == 'kg':
            return self.loss(
                projections, y_label, ref_emb=y_kg_emb
            )
        elif self.name == 'supcon':
            return self.loss(projections, y_label)


class SupConLossSelfDist(SupConLoss):
    def __init__(self, temperature=0.1, **kwargs):
        super().__init__(temperature=temperature, **kwargs)

    def _compute_loss(self, mat, pos_mask, neg_mask):
        if pos_mask.bool().any() and neg_mask.bool().any():

            #############################
            # Adjustment MPR:
            pos_mask.fill_diagonal_(1)  # calc dist between kg and self
            neg_mask.fill_diagonal_(0)
            # Remaining code is the same
            #############################

            # if dealing with actual distances, use negative distances
            if not self.distance.is_inverted:
                mat = -mat
            mat = mat / self.temperature
            mat_max, _ = mat.max(dim=1, keepdim=True)
            mat = mat - mat_max.detach()  # for numerical stability

            denominator = lmu.logsumexp(
                mat, keep_mask=(pos_mask + neg_mask).bool(), add_one=False,
                dim=1
            )
            log_prob = mat - denominator
            mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (
                pos_mask.sum(dim=1) + c_f.small_val(mat.dtype)
            )

            return {
                "loss": {
                    "losses": -mean_log_prob_pos,
                    "indices": c_f.torch_arange_from_size(mat),
                    "reduction_type": "element",
                }
            }
        return self.zero_losses()
