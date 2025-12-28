import torch
import torch.nn as nn
import torch.nn.functional as F


class DockingScorePredictor(nn.Module):
    def __init__(
        self,
        num_protein_atom_types: int,
        num_ligand_atom_types: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        radial_bins: int = 32,
        cutoff: float = 8.0,
    ):
        super().__init__()
        self.cutoff = cutoff

        self.prot_emb = nn.Embedding(num_protein_atom_types, hidden_dim)
        self.lig_emb = nn.Embedding(num_ligand_atom_types, hidden_dim)

        self.radial_centers = nn.Parameter(torch.linspace(0.0, cutoff, radial_bins), requires_grad=False)
        self.radial_width = nn.Parameter(torch.tensor(0.5 * cutoff / radial_bins), requires_grad=False)

        pair_in_dim = hidden_dim * 2 + radial_bins
        mlp_layers = []
        dim = pair_in_dim
        for _ in range(num_layers):
            mlp_layers.append(nn.Linear(dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            dim = hidden_dim
        self.pair_mlp = nn.Sequential(*mlp_layers)

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _radial_basis(self, dist: torch.Tensor) -> torch.Tensor:
        diff = dist.unsqueeze(-1) - self.radial_centers.to(dist.device)
        return torch.exp(-0.5 * (diff / (self.radial_width.to(dist.device) + 1e-8)) ** 2)

    def forward(
        self,
        protein_pos: torch.Tensor,
        protein_atom_type: torch.Tensor,
        protein_batch: torch.Tensor,
        ligand_pos: torch.Tensor,
        ligand_atom_type: torch.Tensor,
        ligand_batch: torch.Tensor,
    ) -> torch.Tensor:
        device = protein_pos.device
        assert protein_pos.shape[-1] == 3 and ligand_pos.shape[-1] == 3

        h_prot = self.prot_emb(protein_atom_type)
        h_lig = self.lig_emb(ligand_atom_type)

        num_complex = int(torch.max(protein_batch.max(), ligand_batch.max()).item()) + 1
        complex_scores = []

        for c in range(num_complex):
            mask_p = (protein_batch == c)
            mask_l = (ligand_batch == c)
            if not (mask_p.any() and mask_l.any()):
                complex_scores.append(torch.zeros(1, device=device))
                continue

            pos_p = protein_pos[mask_p]
            pos_l = ligand_pos[mask_l]
            hp = h_prot[mask_p]
            hl = h_lig[mask_l]

            diff = pos_p.unsqueeze(1) - pos_l.unsqueeze(0)
            dist = torch.linalg.norm(diff, dim=-1)

            mask = (dist < self.cutoff)
            if not mask.any():
                complex_scores.append(torch.zeros(1, device=device))
                continue

            dist_valid = dist[mask]
            rb = self._radial_basis(dist_valid)

            hp_valid = hp.unsqueeze(1).expand(-1, pos_l.size(0), -1)[mask]
            hl_valid = hl.unsqueeze(0).expand(pos_p.size(0), -1, -1)[mask]

            pair_feat = torch.cat([hp_valid, hl_valid, rb], dim=-1)
            pair_hidden = self.pair_mlp(pair_feat)

            complex_repr = pair_hidden.mean(dim=0, keepdim=True)
            score = self.readout(complex_repr)
            complex_scores.append(score.view(1))

        return torch.cat(complex_scores, dim=0)


def docking_guidance_step(
    predictor: DockingScorePredictor,
    batch,
    pred_pos: torch.Tensor,
    step_size: float = 1.0,
    maximize_score: bool = True,
) -> torch.Tensor:
    ligand_pos_in = pred_pos.detach().clone().requires_grad_(True)

    protein_pos = batch['protein_pos']
    protein_atom_type = batch['protein_atom_type']
    protein_batch = batch['protein_batch']
    ligand_atom_type = batch['ligand_atom_type']
    ligand_batch = batch['ligand_batch']

    scores = predictor(
        protein_pos=protein_pos,
        protein_atom_type=protein_atom_type,
        protein_batch=protein_batch,
        ligand_pos=ligand_pos_in,
        ligand_atom_type=ligand_atom_type,
        ligand_batch=ligand_batch,
    )

    target = scores.mean()
    if not maximize_score:
        target = -target

    grad = torch.autograd.grad(target, ligand_pos_in, retain_graph=False, create_graph=False)[0]
    new_pos = ligand_pos_in + step_size * grad
    return new_pos.detach()

