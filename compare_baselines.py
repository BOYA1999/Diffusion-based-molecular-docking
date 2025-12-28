import torch
import pandas as pd
from collections import defaultdict

try:
    from docking_guided_model import DockingGuidedDiffusionModel
    from geometry_guidance import geometry_guidance_step
except ImportError:
    from .docking_guided_model import DockingGuidedDiffusionModel
    from .geometry_guidance import geometry_guidance_step


class BaselineComparator:
    def __init__(
        self,
        base_model,
        docking_config,
        device='cuda',
        num_samples_per_method=20,
        num_test_proteins=10,
    ):
        self.base_model = base_model
        self.docking_config = docking_config
        self.device = device
        self.num_samples_per_method = num_samples_per_method
        self.num_test_proteins = num_test_proteins
        
        self.models = {
            'no_guidance': DockingGuidedDiffusionModel(
                base_model=base_model,
                docking_config={**docking_config, 'use_geometry_guidance': False},
                use_docking_guidance=False,
            ).to(device),
            
            'geometry_only': DockingGuidedDiffusionModel(
                base_model=base_model,
                docking_config={**docking_config, 'use_geometry_guidance': True},
                use_docking_guidance=False,
            ).to(device),
            
            'docking_only': DockingGuidedDiffusionModel(
                base_model=base_model,
                docking_config={**docking_config, 'use_geometry_guidance': False},
                use_docking_guidance=True,
            ).to(device),
            
            'two_stage': DockingGuidedDiffusionModel(
                base_model=base_model,
                docking_config={**docking_config, 'use_geometry_guidance': True},
                use_docking_guidance=True,
            ).to(device),
        }
        
        for model in self.models.values():
            model.eval()
    
    def generate_with_method(self, batch, noiser, method_name):
        model = self.models[method_name]
        
        if method_name == 'no_guidance':
            model.docking_config['use_geometry_guidance'] = False
        elif method_name == 'geometry_only':
            model.docking_config['use_geometry_guidance'] = True
        elif method_name == 'docking_only':
            model.docking_config['use_geometry_guidance'] = False
        else:
            model.docking_config['use_geometry_guidance'] = True
        
        results = model.sample_with_guidance(
            batch=batch,
            noiser=noiser,
            device=self.device,
            num_samples=self.num_samples_per_method,
            top_k=5,
        )
        
        return results
    
    def evaluate_molecules(self, mol_list, protein_path):
        scores = []
        valid_count = 0
        
        for mol_info in mol_list:
            if mol_info is None or mol_info.get('rdmol') is None:
                continue
            
            docking_score = mol_info.get('docking_score', 0.0)
            scores.append({
                'docking_score': docking_score,
                'num_atoms': mol_info.get('rdmol').GetNumAtoms(),
                'valid': True,
            })
            valid_count += 1
        
        if len(scores) == 0:
            return {
                'avg_docking_score': 0.0,
                'best_docking_score': 0.0,
                'success_rate': 0.0,
                'valid_count': 0,
            }
        
        df = pd.DataFrame(scores)
        
        return {
            'avg_docking_score': df['docking_score'].mean(),
            'std_docking_score': df['docking_score'].std(),
            'best_docking_score': df['docking_score'].min(),
            'success_rate': valid_count / len(mol_list) if len(mol_list) > 0 else 0.0,
            'valid_count': valid_count,
            'total_count': len(mol_list),
        }
    
    def compare_all_methods(self, test_batches, noiser):
        results = defaultdict(list)
        
        method_names = ['no_guidance', 'geometry_only', 'docking_only', 'two_stage']
        
        for batch_idx, batch in enumerate(test_batches[:self.num_test_proteins]):
            batch = batch.to(self.device)
            
            for method_name in method_names:
                try:
                    mol_list = self.generate_with_method(batch, noiser, method_name)
                    metrics = self.evaluate_molecules(mol_list, None)
                    metrics['method'] = method_name
                    metrics['batch_idx'] = batch_idx
                    results[method_name].append(metrics)
                except Exception as e:
                    continue
        
        summary = {}
        for method_name in method_names:
            if len(results[method_name]) == 0:
                continue
            
            df = pd.DataFrame(results[method_name])
            summary[method_name] = {
                'avg_docking_score': df['avg_docking_score'].mean(),
                'std_docking_score': df['avg_docking_score'].std(),
                'best_docking_score': df['best_docking_score'].min(),
                'success_rate': df['success_rate'].mean(),
            }
        
        return summary, results
    
    def print_comparison_table(self, summary):
        print("\n" + "=" * 80)
        print("Baseline Comparison Results")
        print("=" * 80)
        
        method_labels = {
            'no_guidance': 'No Guidance',
            'geometry_only': 'Geometry Only',
            'docking_only': 'Docking Only',
            'two_stage': 'Two-Stage (Ours)',
        }
        
        print(f"{'Method':<25} {'Avg Score':<20} {'Best Score':<20} {'Success Rate':<15}")
        print("-" * 80)
        
        method_order = ['no_guidance', 'geometry_only', 'docking_only', 'two_stage']
        for method_name in method_order:
            if method_name not in summary:
                continue
            
            data = summary[method_name]
            print(f"{method_labels[method_name]:<25} "
                  f"{data['avg_docking_score']:>6.2f} ± {data['std_docking_score']:>5.2f}  "
                  f"{data['best_docking_score']:>8.2f}      "
                  f"{data['success_rate']*100:>6.1f}%")
        
        print("-" * 80)

