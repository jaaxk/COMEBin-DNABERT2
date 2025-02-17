import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class EmbeddingNet(nn.Module):
    # Useful code from fast.ai tabular model
    # https://github.com/fastai/fastai/blob/3b7c453cfa3845c6ffc496dd4043c07f3919270e/fastai/tabular/models.py#L6
    def __init__(self, in_sz, out_sz, emb_szs, ps, use_bn=True, actn=nn.ReLU(), pretrained_model=None, cov_model=None, covmodel_notl2normalize=False, llm_embedding_dim=None):
        super(EmbeddingNet, self).__init__()
        self.pretrained_model = pretrained_model
        self.cov_model = cov_model
        self.in_sz = in_sz
        self.out_sz = out_sz
        self.n_embs = len(emb_szs) - 1
        self.covmodel_notl2normalize = covmodel_notl2normalize
        self.llm_embedding_dim = llm_embedding_dim
        self.firstrun = True #DELETE THIS

        if self.llm_embedding_dim is not None:
            self.proj_embedding = nn.Linear(self.llm_embedding_dim, in_sz)
            print(f'Initiallizing embedding projection head with llm_embedding_dim = {self.llm_embedding_dim}')
        else:
            print('WARNING: Not initializing llm projection head')
            self.proj_embedding = None
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Learnable weight for llm : coverage embeddings

        if ps == 0:
            ps = np.zeros(self.n_embs)
        # input layer
        layers = [nn.Linear(self.in_sz, emb_szs[0]), actn]
        # hidden layers
        for i in range(self.n_embs):
            layers += self.bn_drop_lin(
                n_in=emb_szs[i], n_out=emb_szs[i + 1], bn=use_bn, p=ps[i], actn=actn
            )
        # output layer
        layers.append(nn.Linear(emb_szs[-1], self.out_sz))
        self.fc = nn.Sequential(*layers)
        project_layer= [actn, nn.Linear(self.out_sz,self.out_sz)]
        # project_layer= [nn.Linear(self.out_sz,self.out_sz),actn,nn.Linear(self.out_sz,self.out_sz)]
        self.fc2 = nn.Sequential(*project_layer)

    def bn_drop_lin(
            self,
            n_in: int,
            n_out: int,
            bn: bool = True,
            p: float = 0.0,
            actn: nn.Module = None,
    ):
        # https://github.com/fastai/fastai/blob/3b7c453cfa3845c6ffc496dd4043c07f3919270e/fastai/layers.py#L44
        "Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."
        layers = [nn.BatchNorm1d(n_in)] if bn else []
        if p != 0:
            layers.append(nn.Dropout(p))
        layers.append(nn.Linear(n_in, n_out))
        if actn is not None:
            layers.append(actn)
        return layers

    def forward(self, x, x2=None): #x is llm embeddings, x2 is coverage
        if self.pretrained_model is not None and self.cov_model is None:
            kmeremb = self.pretrained_model(x)
            x = torch.cat([F.normalize(self.pretrained_model(x)), x2], dim=-1)
        if self.pretrained_model is not None and self.cov_model is not None:
            kmeremb = self.pretrained_model(x)
            if self.covmodel_notl2normalize:
                x = torch.cat([F.normalize(self.pretrained_model(x)), self.cov_model(x2)], dim=-1)
            else:
                x = torch.cat([F.normalize(self.pretrained_model(x)), F.normalize(self.cov_model(x2))], dim=-1)

        if self.pretrained_model is None and self.cov_model is not None: #This is the path usually taken
            if self.firstrun: #DELETE THIS
                print('CORRECT PATH')
                self.firstrun = False
            x_emb = self.proj_embedding(x) #Scale x down from llm_embedding_dim (768) to outdim_forcov (default 768)
            x2_emb = self.cov_model(x2) #Scale x2 (coverage features) up from cov_dim (~136) to outdim_forcov (dedault 768)
            #Combine embeddings with learnable parameter alpha:
            x = F.normalize((self.alpha * x_emb) + ((1 - self.alpha) * x2_emb))

            del x_emb
            del x2_emb
            torch.cuda.empty_cache()

            """ if self.covmodel_notl2normalize:
                x = torch.cat([x, self.cov_model(x2)], dim=-1) #should be size [rows x (llm_dim + outdim_forcov)]
            else:
                x = torch.cat([x, F.normalize(self.cov_model(x2))], dim=-1) """

        output = self.fc(x)

        if self.cov_model is not None and self.pretrained_model is not None:
            return output, self.cov_model(x2), kmeremb
        elif self.cov_model is not None and self.pretrained_model is None:
            return output, self.cov_model(x2)
        else:
            return output#F.normalize(output) #output #/ torch.linalg.vector_norm(output, dim=-1, keepdim=True)
