import torch
import torch.nn.functional as F


def _center_and_clash_energy(
    ligand_pos: torch.Tensor,
    batch_node: torch.Tensor,
    pocket_center: torch.Tensor,
    center_radius: float = 5.0,
    clash_distance: float = 1.2,
    clash_weight: float = 1.0,
) -> torch.Tensor:
    device = ligand_pos.device
    num_graphs = int(batch_node.max().item()) + 1 if ligand_pos.numel() > 0 else 0
    total_energy = ligand_pos.new_zeros(())

    if num_graphs == 0:
        return total_energy

    for g in range(num_graphs):
        mask = (batch_node == g)
        if not torch.any(mask):
            continue

        pos_g = ligand_pos[mask]
        center_g = pocket_center[g].to(device).unsqueeze(0)

        d = torch.linalg.norm(pos_g - center_g, dim=-1)
        outside = F.relu(d - center_radius)
        center_energy = (outside ** 2).mean()

        if pos_g.size(0) > 1:
            diff = pos_g.unsqueeze(0) - pos_g.unsqueeze(1)
            dist = torch.linalg.norm(diff + torch.eye(pos_g.size(0), device=device).unsqueeze(-1), dim=-1)
            iu = torch.triu_indices(pos_g.size(0), pos_g.size(0), offset=1)
            pair_dist = dist[iu[0], iu[1]]
            clash = F.relu(clash_distance - pair_dist)
            clash_energy = (clash ** 2).mean()
        else:
            clash_energy = ligand_pos.new_zeros(())

        total_energy = total_energy + center_energy + clash_weight * clash_energy

    return total_energy


def geometry_guidance_step(
    batch,
    outputs: dict,
    center_radius: float = 5.0,
    clash_distance: float = 1.2,
    clash_weight: float = 1.0,
    step_size: float = 1.0,
) -> torch.Tensor:
    if 'pred_pos' not in outputs:
        return outputs.get('pred_pos', None)

    ligand_pos = outputs['pred_pos']
    if ligand_pos is None or ligand_pos.numel() == 0:
        return ligand_pos

    batch_node = batch.get('node_type_batch', None)
    pocket_center = batch.get('pocket_center', None)
    if batch_node is None or pocket_center is None:
        return ligand_pos

    ligand_pos_in = ligand_pos.detach().clone().requires_grad_(True)

    energy = _center_and_clash_energy(
        ligand_pos_in,
        batch_node=batch_node,
        pocket_center=pocket_center,
        center_radius=center_radius,
        clash_distance=clash_distance,
        clash_weight=clash_weight,
    )

    if not torch.isfinite(energy):
        return ligand_pos

    grad = torch.autograd.grad(energy, ligand_pos_in, retain_graph=False, create_graph=False)[0]
    if grad is None:
        return ligand_pos

    new_pos = ligand_pos_in - step_size * grad
    return new_pos.detach()

