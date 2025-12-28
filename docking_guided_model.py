import torch
import torch.nn as nn
from torch.nn import Module
from easydict import EasyDict
import os
from typing import Optional, Dict

try:
    from geometry_guidance import geometry_guidance_step
except ImportError:
    from .geometry_guidance import geometry_guidance_step


class DockingGuidedDiffusionModel(Module):
    def __init__(
        self,
        base_model,
        docking_config: Dict,
        use_docking_guidance: bool = True,
        guidance_scale: float = 1.0,
        docking_method: str = 'vina',
    ):
        super().__init__()
        self.base_model = base_model
        self.docking_config = EasyDict(docking_config)
        self.use_docking_guidance = use_docking_guidance
        self.guidance_scale = guidance_scale
        self.docking_method = docking_method
        self.tmp_dir = self.docking_config.get('tmp_dir', './tmp_docking')
        os.makedirs(self.tmp_dir, exist_ok=True)
        
    def forward(self, batch, **kwargs):
        outputs = self.base_model(batch, **kwargs)
        
        if self.use_docking_guidance and self.training:
            docking_scores = self._compute_docking_scores(batch, outputs)
            outputs['docking_scores'] = docking_scores
            outputs['docking_loss'] = -torch.mean(docking_scores)
        
        return outputs
    
    def sample_with_guidance(
        self,
        batch,
        noiser,
        device='cuda',
        num_samples: int = 1,
        top_k: int = 5,
        progress_callback=None
    ):
        self.eval()
        all_results = []
        total_tasks = num_samples * batch.num_graphs
        
        with torch.no_grad():
            for sample_idx in range(num_samples):
                if progress_callback:
                    progress_callback(
                        sample_idx * batch.num_graphs,
                        total_tasks,
                        f"Generating sample {sample_idx + 1}/{num_samples}"
                    )
                
                batch_sample = batch.clone() if hasattr(batch, 'clone') else batch
                
                # Import sample_loop3 from base model or user-provided sampling function
                # Users should provide their own sampling function via docking_config
                sample_func = self.docking_config.get('sample_function', None)
                if sample_func is None:
                    try:
                        from models.sample import sample_loop3 as sample_func
                    except ImportError:
                        raise ImportError(
                            "Please provide 'sample_function' in docking_config or "
                            "ensure 'models.sample.sample_loop3' is available"
                        )
                
                batch_sample, outputs, trajs = sample_func(
                    batch_sample, self.base_model, noiser, device, is_ar='', off_tqdm=False
                )

                if self.docking_config.get('use_geometry_guidance', False):
                    step_size = self.docking_config.get('geometry_guidance_step_size', 1.0)
                    center_radius = self.docking_config.get('geometry_center_radius', 5.0)
                    clash_distance = self.docking_config.get('geometry_clash_distance', 1.2)
                    clash_weight = self.docking_config.get('geometry_clash_weight', 1.0)

                    new_pos = geometry_guidance_step(
                        batch_sample,
                        outputs,
                        center_radius=center_radius,
                        clash_distance=clash_distance,
                        clash_weight=clash_weight,
                        step_size=step_size,
                    )
                    if new_pos is not None:
                        outputs['pred_pos'] = new_pos
                
                mol_info_list = self._decode_outputs(batch_sample, outputs)
                
                for mol_info in mol_info_list:
                    if mol_info is None:
                        continue
                    all_results.append(mol_info)
        
        all_results.sort(key=lambda x: x.get('docking_score', -999.0), reverse=True)
        return all_results[:top_k] if len(all_results) > top_k else all_results
    
    def _compute_docking_scores(self, batch, outputs) -> Optional[torch.Tensor]:
        return None
    
    def _decode_outputs(self, batch, outputs):
        return []

