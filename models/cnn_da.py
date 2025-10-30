import torch
import torch.nn as nn
from torchsummary import summary

from torch.autograd import Function

from sklearn.metrics import accuracy_score


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer.
    This layer is used to reverse the gradient during backpropagation
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class MMDLoss(nn.Module):

    def __init__(self, bandwith):
        super().__init__()
        self.bandwith = bandwith

    def kernel(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        nom = -L2_distances[None, ...]
        denom = (2 * self.bandwith ** 2)  # [:, None, None]
        return torch.exp(nom / denom).sum(dim=0)

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY


class CnnDA(nn.Module):

    def __init__(
            self, n_classes: int, n_domains: int = None, dataset: str = None,
            da: False = True, mmd: bool = False
    ):
        super().__init__()

        self.da = da  # Domain adversarial training
        self.mmd = mmd  # MMD loss

        # Select the correct embedding network based on the dataset
        # TODO
        if dataset == 'CWRU':
            self.conv, self.dom_clf, self.emb_clf, self.clf = \
                cwru_architecture(
                    n_classes, n_domains, conv_out_dim=96, input_channels=1
                )
            self.domain = 'rpm'
        elif dataset == 'PHM22Rockdrill':
            self.conv, self.dom_clf, self.emb_clf, self.clf = \
                phm22_rockdrill_architecture(n_classes, n_domains)
        elif dataset == 'PHMAP23Spacecraftpropulsion':
            self.conv, self.dom_clf, self.emb_clf, self.clf = \
                phmap23_spacecraftpropulsion_architecture(
                    n_classes, n_domains, conv_out_dim=40, input_channels=7
                )
        else:
            raise NotImplementedError(
                f'Dataset {dataset} not implemented.'
            )

    def forward(
            self, input, domain_adversarial: bool = False,
    ) -> torch.Tensor:
        features = self.conv(input)
        out = {'clf_scores': self.clf(self.emb_clf(features))}
        if domain_adversarial:
            grl_features = GradientReversalFunction.apply(features, 1.0)
            out['dom_scores'] = self.dom_clf(grl_features)
        return out

    def get_summary(self, input):
        device = next(self.parameters()).device
        summary(
            self, (input[0].shape[0], input[0].shape[1]),
            device=device.type
        )

    def train_epoch_step(
            self, data_loader, optimizer, criterion, mode=None
    ):
        # mode is ignored in DA setting
        self.train()
        device = next(self.parameters()).device
        for data in data_loader:
            data = {k: v.to(device) for k, v in data.items()}
            optimizer.zero_grad()
            out = self(data['X'], self.da)
            loss = criterion(out['clf_scores'], data['y'])
            if self.da:
                criterion_da = nn.CrossEntropyLoss()
                loss += criterion_da(
                    out['dom_scores'], data['y_domain']
                )
            if self.mmd:
                raise NotImplementedError("MMD not implemented yet.")
            loss.backward()
            optimizer.step()

    def eval_epoch_step(self, data_loader, criterion, mode=None):
        # mode is ignored in DA setting
        self.eval()
        device = next(self.parameters()).device
        acc = []
        with torch.no_grad():
            for data in data_loader:
                data = {k: v.to(device) for k, v in data.items()}
                out = self(data['X'])['clf_scores']
                loss = criterion(out, data['y'])
                acc_d = accuracy_score(
                    data['y'].to('cpu'),
                    out.to('cpu').argmax(dim=1)
                )
                acc.append(acc_d * len(data['y']))
            acc = sum(acc) / len(data_loader.dataset)
            return loss, acc

    def get_proj_emb_clf_y_from_dataloader(
            self, data_loader, device, projection=False
    ):
        # Needed for evaluation compatibility with other models
        # projection is ignored in DA setting
        self.eval()
        with torch.no_grad():
            emb_X = []
            proj_X = []
            clf_X = []
            y = []
            for data in data_loader:
                features = self.conv(data['X'].to(device))
                emb = self.emb_clf(features)
                emb_X.append(emb.to('cpu'))
                clf = self.clf(emb)
                clf_X.append(clf.to('cpu'))
                y.append(data['y'].to('cpu'))
            emb_X = torch.cat(emb_X, dim=0).detach().numpy()
            proj_X = None
            clf_X = torch.cat(clf_X, dim=0).detach().numpy()
            y = torch.cat(y, dim=0).detach().numpy()
        return proj_X, emb_X, clf_X, y


def cwru_architecture(n_classes, n_domains, conv_out_dim=96, input_channels=1):
    kernel = 12
    conv = nn.Sequential(
        nn.Conv1d(input_channels, 64, kernel),
        nn.LeakyReLU(0.5),
        nn.MaxPool1d(kernel_size=kernel, stride=2),
        nn.Dropout1d(p=0.1),
        nn.Conv1d(64, 32, kernel),
        nn.LeakyReLU(0.5),
        nn.MaxPool1d(kernel_size=kernel, stride=2),
        nn.Dropout1d(p=0.1),
        nn.Conv1d(32, 16, kernel),
        nn.LeakyReLU(0.5),
        nn.MaxPool1d(kernel_size=kernel, stride=2),
        nn.Dropout1d(p=0.1),
        nn.Conv1d(16, 8, kernel),
        nn.LeakyReLU(0.5),
        nn.MaxPool1d(kernel_size=kernel, stride=2),
        nn.Dropout1d(p=0.1),
        nn.Flatten()
    )

    dom_clf = nn.Sequential(
        nn.Linear(conv_out_dim, 64),
        nn.LeakyReLU(0.5),
        # Added for DA
        nn.Linear(64, 32),
        nn.LeakyReLU(0.5),
        ################
        nn.Linear(32, n_domains)  # 32 -> 64 changed for DA
    )

    emb_clf = nn.Sequential(
        nn.Linear(conv_out_dim, 64),
        nn.LeakyReLU(0.5),
        # Added for DA
        nn.Linear(64, 32),
        nn.LeakyReLU(0.5),
        ################
    )

    clf = nn.Linear(32, n_classes)  # 32 -> 64 changed for DA

    return conv, dom_clf, emb_clf, clf


def phm22_rockdrill_architecture(n_classes, n_domains):
    conv = nn.Sequential(
        nn.Conv1d(
            3, 16, kernel_size=15, stride=1,
            padding=7,
        ),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.Conv1d(
            16, 32, kernel_size=15, stride=1,
            padding=7,
        ),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2),
        nn.Conv1d(
            32, 64, kernel_size=7, stride=1, padding=3
        ),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Conv1d(
            64, 128, kernel_size=7, stride=1, padding=3,
        ),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2),
        nn.Conv1d(
            128, 256, kernel_size=3, stride=1, padding=1,
        ),
        nn.Flatten(),
        nn.Linear(57600, 512),
        nn.ReLU()
    )

    dom_clf = nn.Sequential(
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Linear(64, n_domains)
    )

    emb_clf = nn.Sequential(
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.BatchNorm1d(64),
        nn.ReLU()
    )

    clf = nn.Linear(64, n_classes)

    return conv, dom_clf, emb_clf, clf


def phmap23_spacecraftpropulsion_architecture(
        n_classes,
        n_domains,
        conv_out_dim=40,
        input_channels=7
):
    conv, dom_clf, emb_clf, clf = cwru_architecture(
        n_classes, n_domains, conv_out_dim, input_channels
    )
    return conv, dom_clf, emb_clf, clf
