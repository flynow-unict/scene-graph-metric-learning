"""Loss per metric learning: Triplet Margin e NT-Xent."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def triplet_loss(anchor, positive, negative, margin: float = 0.2):
    return F.triplet_margin_loss(anchor, positive, negative, margin=margin, p=2)


def batch_hard_triplet_loss(anchor, positive, margin: float = 0.2, mining: str = "semi"):
    """Triplet con negativi in-batch (coseno), hard oppure semi-hard.

    anchor/positive: [B, D] gia' L2-normalizzati. Il positivo di ogni ancora e' la
    sua vista aumentata (diagonale della matrice di similarita'); i negativi sono
    gli altri elementi del batch.

    - mining="hard": negativo = il piu' simile in assoluto (max coseno). Aggressivo,
      tende a scegliere negativi degeneri e a far collassare la rappresentazione.
    - mining="semi": negativo semi-hard (FaceNet) = tra quelli MENO simili del
      positivo (sim < pos_sim, quindi gia' dal lato giusto) si prende il piu' vicino.
      Da' un gradiente informativo ma non degenere; dove non esistono semi-hard si
      ricade sull'hard negative.
    """
    B = anchor.size(0)
    sim = anchor @ positive.t()                       # [B, B] coseni
    pos_sim = sim.diag()                              # ancora . positivo
    eye = torch.eye(B, dtype=torch.bool, device=sim.device)
    neg = sim.masked_fill(eye, float("-inf"))         # esclude il positivo

    if mining == "semi":
        semi = neg.masked_fill(neg >= pos_sim.unsqueeze(1), float("-inf"))
        neg_sim = semi.max(dim=1).values
        no_semi = torch.isinf(neg_sim)                # nessun semi-hard -> hard negative
        neg_sim = torch.where(no_semi, neg.max(dim=1).values, neg_sim)
    else:
        neg_sim = neg.max(dim=1).values

    # distanza coseno d = 1 - sim  ->  loss = relu(d_ap - d_an + m) = relu(neg - pos + m)
    return torch.relu(neg_sim - pos_sim + margin).mean()


class NTXentLoss(nn.Module):
    """NT-Xent (SimCLR): due viste z1, z2 [B, D] degli stessi B elementi."""

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        B = z1.size(0)
        z = F.normalize(torch.cat([z1, z2], dim=0), dim=-1)   # [2B, D]
        sim = z @ z.t() / self.temperature                    # [2B, 2B]

        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float("-inf"))

        targets = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
        return F.cross_entropy(sim, targets)
