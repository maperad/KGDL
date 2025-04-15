import torch
from torch.nn import functional as F
from torch import nn

from torch_geometric.data import HeteroData

from sklearn.metrics import accuracy_score


class FinalLinearLayer(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.lin(x)


class BaseModel(torch.nn.Module):

    def __init__(
            self, n_classes: int, embedding_dim: int,
            projection_dim: int = None, parts: list = ['clf']
    ):
        super().__init__()
        self.n_classes = n_classes
        self.projection_dim = projection_dim
        self.embedding_dim = embedding_dim
        self.parts = parts

        self.net_embed = None

        if self.projection_dim is not None:
            # Embedding output (only for contrastive loss)
            self.net_project = FinalLinearLayer(
                self.embedding_dim, self.projection_dim
            )

        self.net_clf = FinalLinearLayer(self.embedding_dim, n_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.net_embed(x)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net_project(self.embed(x)), p=2)

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        return self.net_clf(self.embed(x))

    def forward(self, input, mode: str = 'clf') -> torch.Tensor:
        x_emb = self.embed(input)
        if mode == 'embedding':
            return (
                F.normalize(self.net_project(x_emb), p=2)
            )
        elif mode == 'clf':
            return self.net_clf(x_emb)

    def get_summary(self, input):
        raise NotImplementedError

    def train_epoch_step(
            self, data_loader, optimizer,
            criterion, mode
    ):
        if ('embedding' in self.parts) and (mode == 'clf'):
            # Freeze embedding parts -> no gradient update
            # Set to eval for batch norm layers -> no mean/std update
            self.net_embed.eval()
            self.net_project.eval()
            for param in self.net_embed.parameters():
                param.requires_grad = False
            for param in self.net_project.parameters():
                param.requires_grad = False
            # Ensure clf part is in train mode
            self.net_clf.train()
            for param in self.net_clf.parameters():
                param.requires_grad = True
            emb_clf = True
        else:
            # Ensure all parts are in train mode
            self.net_embed.train()
            if self.projection_dim is not None:
                self.net_project.train()
            self.net_clf.train()
            for param in self.parameters():
                param.requires_grad = True
            emb_clf = False
        device = next(self.parameters()).device
        for data in data_loader:
            # Check if batch size is 1
            # -> skip batch because of batch norm
            if isinstance(data, dict):
                if len(next(iter(data.values()))) == 1:
                    continue
            if isinstance(data, HeteroData):
                if len(data) == 1:
                    continue
            optimizer.zero_grad()
            if emb_clf:
                loss, _ = self._epoch_step_body_embclf(
                    data, device, criterion
                )
            else:
                loss, _, _ = self._epoch_step_body(
                    data, device, mode, criterion
                )
            loss.backward()
            optimizer.step()

    def eval_epoch_step(
            self, data_loader, criterion, mode
    ):
        self.eval()
        device = next(self.parameters()).device
        acc = []
        if ('embedding' in self.parts) and (mode == 'clf'):
            emb_clf = True
        else:
            emb_clf = False
        with torch.no_grad():
            for data in data_loader:
                if emb_clf:
                    loss, out_clf = self._epoch_step_body_embclf(
                        data, device, criterion
                    )
                else:
                    loss, out_clf, _ = self._epoch_step_body(
                        data, device, mode, criterion
                    )
                if mode == 'clf':
                    acc_d = accuracy_score(
                        data['y'].to('cpu'), out_clf.to('cpu').argmax(dim=1)
                    )
                    acc.append(acc_d * len(data['y']))

            if mode == 'clf':
                acc = sum(acc) / len(data_loader.dataset)
            else:
                acc = None
            return loss, acc

    def _epoch_step_body(
            self, data, device, mode, criterion
    ):
        raise NotImplementedError

    def _epoch_step_body_embclf(
            self, data, device, criterion
    ):
        data = {k: v.to(device) for k, v in data.items()}
        out_clf = self.net_clf(data['X'])
        loss = criterion(out_clf, data['y'])
        return loss, out_clf

    def get_proj_emb_clf_y_from_dataloader(
            self, data_loader, device, projection=True
    ):
        self.eval()
        with torch.no_grad():
            emb_X = []
            proj_X = []
            clf_X = []
            y = []
            for data in data_loader:
                if isinstance(data, HeteroData):
                    emb = self.embed(data.to(device))
                if isinstance(data, dict):
                    emb = self.embed(data['X'].to(device))
                emb_X.append(emb.to('cpu'))
                if projection:
                    proj = F.normalize(self.net_project(emb), p=2)
                    proj_X.append(proj.to('cpu'))
                clf = self.net_clf(emb)
                clf_X.append(clf.to('cpu'))
                y.append(data['y'].to('cpu'))
            emb_X = torch.cat(emb_X, dim=0).detach().numpy()
            if projection:
                proj_X = torch.cat(proj_X, dim=0).detach().numpy()
            else:
                proj_X = None
            clf_X = torch.cat(clf_X, dim=0).detach().numpy()
            y = torch.cat(y, dim=0).detach().numpy()
        return proj_X, emb_X, clf_X, y
