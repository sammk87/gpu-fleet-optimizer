import json
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Union
from enum import Enum

class WorkloadType(Enum):
    TRAINING = "training"
    INFERENCE = "inference"
    FINE_TUNING = "fine_tuning"

class ModelSize(Enum):
    SMALL = "small"      # 1-13B parameters
    MEDIUM = "medium"    # 14-70B parameters  
    LARGE = "large"      # 70B+ parameters
    MASSIVE = "massive"  # 100B+ parameters

@dataclass
class GPUSpec:
    name: str
    family: str
    gpu_type: str
    vcpus: int
    gpu_memory_gb: int
    gpus_per_instance: int
    compute_tflops: float
    interconnect_type: str
    interconnect_bandwidth_gbps: int
    on_demand_cost_per_hour: float
    spot_cost_per_hour: float
    memory_bandwidth_factor: float
    interconnect_multiplier: float
    description: str = ""

class AWSGPUMigrationAnalyzer:
    def __init__(self, config_dir: str = "config"):
        """
        Initialize the analyzer with configuration files
        
        Args:
            config_dir: Directory containing configuration JSON files
        """
        self.config_dir = config_dir
        self.gpu_config = self._load_gpu_config()
        self.model_config = self._load_model_config()
        self.workloads_config = self._load_workloads_config()
        self.gpu_specs = self._initialize_gpu_specs()
    
    def _load_gpu_config(self) -> Dict:
        """Load GPU instance configuration from JSON file"""
        config_path = os.path.join(self.config_dir, "gpu_instances.json")
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"GPU configuration file not found: {config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in GPU configuration file: {e}")
    
    def _load_model_config(self) -> Dict:
        """Load model and workload configuration from JSON file"""
        config_path = os.path.join(self.config_dir, "model_config.json")
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model configuration file not found: {config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in model configuration file: {e}")
    
    def _load_workloads_config(self) -> Dict:
        """Load workloads configuration from JSON file"""
        config_path = os.path.join(self.config_dir, "workloads.json")
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Workloads config is optional
            print(f"⚠️  Workloads configuration file not found: {config_path}")
            return {"workloads": {}, "workload_sets": {}}
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in workloads configuration file: {e}")
    
    def _initialize_gpu_specs(self) -> Dict[str, GPUSpec]:
        """Initialize GPU specifications from JSON configuration"""
        specs = {}
        
        for instance_id, config in self.gpu_config["gpu_instances"].items():
            specs[instance_id] = GPUSpec(
                name=config["name"],
                family=config["family"],
                gpu_type=config["gpu_type"],
                vcpus=config["vcpus"],
                gpu_memory_gb=config["gpu_memory_gb"],
                gpus_per_instance=config["gpus_per_instance"],
                compute_tflops=config["compute_tflops"],
                interconnect_type=config["interconnect_type"],
                interconnect_bandwidth_gbps=config["interconnect_bandwidth_gbps"],
                on_demand_cost_per_hour=config["on_demand_cost_per_hour"],
                spot_cost_per_hour=config["spot_cost_per_hour"],
                memory_bandwidth_factor=config["memory_bandwidth_factor"],
                interconnect_multiplier=config["interconnect_multiplier"],
                description=config.get("description", "")
            )
        
        return specs

    def calculate_training_efficiency(self, spec: GPUSpec, model_size_gb: float, use_spot: bool = False) -> float:
        """Calculate training efficiency score"""
        cost = spec.spot_cost_per_hour if use_spot else spec.on_demand_cost_per_hour
        
        # Memory efficiency - how well GPU memory matches model requirements
        memory_efficiency = min(1.0, spec.gpu_memory_gb / model_size_gb) * spec.memory_bandwidth_factor
        
        # Compute efficiency per dollar
        compute_per_dollar = spec.compute_tflops / cost
        
        # Account for interconnect for multi-GPU training
        total_efficiency = compute_per_dollar * memory_efficiency * spec.interconnect_multiplier
        
        return total_efficiency

    def calculate_inference_efficiency(self, spec: GPUSpec, model_size_gb: float, use_spot: bool = False) -> float:
        """Calculate inference efficiency score"""
        cost = spec.spot_cost_per_hour if use_spot else spec.on_demand_cost_per_hour
        
        # Available memory for batching
        available_memory = max(0, spec.gpu_memory_gb - model_size_gb)
        batch_capacity = available_memory / 2  # Rough estimate for context/batch overhead
        
        # Estimated tokens per second (simplified formula)
        tokens_per_second = (spec.compute_tflops / 100) * min(1.0, batch_capacity / 10)
        
        # Efficiency = throughput per dollar
        efficiency = tokens_per_second / cost
        
        return efficiency

    def get_model_size_gb(self, model_size: ModelSize, precision: str = "fp16") -> float:
        """Estimate model size in GB based on parameter count and precision from config"""
        # Get model configuration
        model_key = model_size.value
        if model_key not in self.model_config["model_sizes"]:
            raise ValueError(f"Unknown model size: {model_key}")
        
        model_info = self.model_config["model_sizes"][model_key]
        parameter_count_billion = model_info["parameter_count_billion"]
        
        # Get precision configuration
        if precision not in self.model_config["precision_formats"]:
            raise ValueError(f"Unknown precision format: {precision}")
        
        precision_info = self.model_config["precision_formats"][precision]
        bytes_per_parameter = precision_info["bytes_per_parameter"]
        
        # Calculate model size in GB
        return parameter_count_billion * bytes_per_parameter

    def can_fit_model(self, spec: GPUSpec, model_size_gb: float, gpu_count: int = 1) -> Tuple[bool, str]:
        """
        Check if a model can fit in the specified GPU configuration
        Returns (can_fit, reason)
        """
        total_memory = spec.gpu_memory_gb * gpu_count
        required_memory = model_size_gb * 1.5  # 1.5x for overhead (optimizer states, gradients, etc.)
        
        if total_memory < model_size_gb:
            return False, f"Insufficient memory: {total_memory}GB available, {model_size_gb}GB required for model"
        
        if total_memory < required_memory:
            return False, f"Insufficient memory for training overhead: {total_memory}GB available, {required_memory:.1f}GB required"
        
        return True, ""

    def analyze_migration(self, 
                         current_instance: str,
                         current_gpu_count: int,
                         workload_type: WorkloadType,
                         model_size: ModelSize,
                         monthly_hours: int = 720,
                         use_spot: bool = False,
                         precision: str = "fp16") -> List[Dict]:
        """
        Analyze migration options from current setup
        """
        if current_instance not in self.gpu_specs:
            raise ValueError(f"Unknown instance type: {current_instance}")
        
        current_spec = self.gpu_specs[current_instance]
        model_size_gb = self.get_model_size_gb(model_size, precision)
        
        # Calculate current performance and cost
        if workload_type == WorkloadType.TRAINING or workload_type == WorkloadType.FINE_TUNING:
            current_efficiency = self.calculate_training_efficiency(current_spec, model_size_gb, use_spot)
        else:
            current_efficiency = self.calculate_inference_efficiency(current_spec, model_size_gb, use_spot)
        
        current_cost = (current_spec.spot_cost_per_hour if use_spot else current_spec.on_demand_cost_per_hour)
        current_monthly_cost = current_cost * current_gpu_count * monthly_hours
        
        # Analyze all migration options
        migration_options = []
        
        for instance_name, spec in self.gpu_specs.items():
            if instance_name == current_instance:
                continue
            
            # First, check if we can fit the model with a single GPU
            can_fit_single, fit_reason = self.can_fit_model(spec, model_size_gb, 1)
            
            if not can_fit_single:
                # Try with multiple GPUs (up to max available per instance)
                min_gpus_needed = max(1, int((model_size_gb * 1.5) / spec.gpu_memory_gb) + 1)
                if min_gpus_needed <= spec.gpus_per_instance:
                    can_fit_multi, _ = self.can_fit_model(spec, model_size_gb, min_gpus_needed)
                    if can_fit_multi:
                        # Model can fit with multiple GPUs
                        optimal_gpu_count = min_gpus_needed
                        can_fit = True
                        fit_reason = ""
                    else:
                        can_fit = False
                else:
                    can_fit = False
            else:
                can_fit = True
                optimal_gpu_count = 1
            
            if not can_fit:
                # Cannot fit the model - add to results with clear indication
                migration_options.append({
                    "target_instance": instance_name,
                    "target_spec": spec.name,
                    "gpu_type": spec.gpu_type,
                    "optimal_gpu_count": "N/A",
                    "monthly_cost": "N/A",
                    "cost_savings_monthly": "N/A",
                    "cost_savings_percent": "N/A",
                    "performance_improvement_percent": "N/A",
                    "efficiency_score": "N/A",
                    "memory_sufficient": False,
                    "total_memory_gb": spec.gpu_memory_gb * spec.gpus_per_instance,
                    "recommendation": f"CANNOT FIT MODEL - {fit_reason}",
                    "risk_level": "N/A"
                })
                continue
            
            # Calculate efficiency for this option (only for viable options)
            if workload_type == WorkloadType.TRAINING or workload_type == WorkloadType.FINE_TUNING:
                new_efficiency = self.calculate_training_efficiency(spec, model_size_gb, use_spot)
            else:
                new_efficiency = self.calculate_inference_efficiency(spec, model_size_gb, use_spot)
            
            # Determine optimal GPU count for equivalent performance (if not already set above)
            if can_fit_single and current_efficiency > 0 and new_efficiency > 0:
                performance_ratio = new_efficiency / current_efficiency
                optimal_gpu_count = max(1, int(current_gpu_count / performance_ratio))
                # Ensure we don't exceed instance capacity
                optimal_gpu_count = min(optimal_gpu_count, spec.gpus_per_instance)
                
                # Re-check memory with optimal count
                can_fit_optimal, _ = self.can_fit_model(spec, model_size_gb, optimal_gpu_count)
                if not can_fit_optimal:
                    # Fall back to minimum required GPUs
                    optimal_gpu_count = max(1, int((model_size_gb * 1.5) / spec.gpu_memory_gb) + 1)
                    optimal_gpu_count = min(optimal_gpu_count, spec.gpus_per_instance)
            
            # Calculate costs
            new_cost_per_hour = (spec.spot_cost_per_hour if use_spot else spec.on_demand_cost_per_hour)
            new_monthly_cost = new_cost_per_hour * optimal_gpu_count * monthly_hours
            
            # Final memory check
            total_memory = spec.gpu_memory_gb * optimal_gpu_count
            memory_sufficient = total_memory >= model_size_gb * 1.5  # 1.5x for overhead
            
            # Calculate savings and performance metrics
            cost_savings_monthly = current_monthly_cost - new_monthly_cost
            cost_savings_percent = (cost_savings_monthly / current_monthly_cost) * 100
            
            if current_efficiency > 0 and current_gpu_count > 0:
                performance_improvement = ((new_efficiency * optimal_gpu_count) / 
                                         (current_efficiency * current_gpu_count) - 1) * 100
            else:
                performance_improvement = 0
            
            # Determine recommendation strength
            recommendation = self._get_recommendation(
                spec, workload_type, model_size, cost_savings_percent, 
                performance_improvement, memory_sufficient
            )
            
            migration_options.append({
                "target_instance": instance_name,
                "target_spec": spec.name,
                "gpu_type": spec.gpu_type,
                "optimal_gpu_count": optimal_gpu_count,
                "monthly_cost": new_monthly_cost,
                "cost_savings_monthly": cost_savings_monthly,
                "cost_savings_percent": round(cost_savings_percent, 1),
                "performance_improvement_percent": round(performance_improvement, 1),
                "efficiency_score": round(new_efficiency, 3),
                "memory_sufficient": memory_sufficient,
                "total_memory_gb": total_memory,
                "recommendation": recommendation,
                "risk_level": self._assess_risk(spec, model_size, workload_type)
            })
        
        # Sort by efficiency score descending (handle "N/A" values)
        migration_options.sort(key=lambda x: x["efficiency_score"] if x["efficiency_score"] != "N/A" else -1, reverse=True)
        
        return {
            "current_setup": {
                "instance": current_instance,
                "gpu_count": current_gpu_count,
                "monthly_cost": current_monthly_cost,
                "efficiency_score": round(current_efficiency, 3)
            },
            "migration_options": migration_options[:5]  # Top 5 options
        }

    def _get_recommendation(self, spec: GPUSpec, workload: WorkloadType, 
                           model_size: ModelSize, cost_savings: float, 
                           performance_gain: float, memory_ok: bool) -> str:
        """Determine recommendation strength"""
        if not memory_ok:
            return "NOT_RECOMMENDED - Insufficient Memory"
        
        if workload == WorkloadType.TRAINING or workload == WorkloadType.FINE_TUNING:
            if model_size == ModelSize.LARGE or model_size == ModelSize.MASSIVE:
                if spec.family == "G":
                    return "NOT_RECOMMENDED - Use P-series for large model training"
                elif performance_gain > 30:
                    return "HIGHLY_RECOMMENDED - Better performance + efficiency"
                elif cost_savings > 20:
                    return "RECOMMENDED - Cost optimization"
            else:
                if cost_savings > 40:
                    return "HIGHLY_RECOMMENDED - Significant cost savings"
                elif performance_gain > 50:
                    return "RECOMMENDED - Performance upgrade"
        
        else:  # Inference
            if cost_savings > 50:
                return "HIGHLY_RECOMMENDED - Major cost reduction"
            elif cost_savings > 20 and performance_gain > 0:
                return "RECOMMENDED - Good cost/performance balance"
            elif performance_gain > 100:
                return "RECOMMENDED - Significant performance boost"
        
        if cost_savings > 0 and performance_gain > 0:
            return "CONSIDER - Moderate improvement"
        
        return "NOT_RECOMMENDED - No clear benefit"

    def _assess_risk(self, spec: GPUSpec, model_size: ModelSize, workload: WorkloadType) -> str:
        """Assess migration risk level"""
        if spec.name.startswith("P6"):
            return "HIGH - New generation, limited availability"
        elif spec.name.startswith("G6e"):
            return "MEDIUM - Newer instance, check regional availability"  
        elif spec.family == "G" and (model_size == ModelSize.LARGE or model_size == ModelSize.MASSIVE):
            return "HIGH - G-series may not handle large models well"
        elif spec.name.startswith("G4"):
            return "MEDIUM - Older generation, consider lifecycle"
        else:
            return "LOW - Established instance type"

    def generate_migration_report(self, current_instance: str, current_gpu_count: int,
                                workload_type: WorkloadType, model_size: ModelSize,
                                monthly_hours: int = 720, use_spot: bool = False) -> str:
        """Generate a comprehensive migration report"""
        
        analysis = self.analyze_migration(
            current_instance, current_gpu_count, workload_type, 
            model_size, monthly_hours, use_spot
        )
        
        report = f"""
AWS GPU INSTANCE MIGRATION ANALYSIS REPORT
==========================================

Current Setup:
- Instance: {current_instance} ({analysis['current_setup']['gpu_count']} GPUs)
- Monthly Cost: ${analysis['current_setup']['monthly_cost']:,.2f}
- Efficiency Score: {analysis['current_setup']['efficiency_score']}
- Workload: {workload_type.value.title()}
- Model Size: {model_size.value.title()} (~{self.get_model_size_gb(model_size):.0f}GB)
- Using Spot: {'Yes' if use_spot else 'No'}

TOP MIGRATION RECOMMENDATIONS:
"""
        
        for i, option in enumerate(analysis['migration_options'][:3]):
            # Handle "N/A" values for options that cannot fit the model
            if option['cost_savings_percent'] == "N/A":
                savings_indicator = ""
                performance_indicator = ""
                monthly_cost_str = "N/A"
                cost_savings_str = "N/A"
                performance_str = "N/A"
            else:
                savings_indicator = "💰" if option['cost_savings_percent'] > 20 else ""
                performance_indicator = "🚀" if option['performance_improvement_percent'] > 30 else ""
                monthly_cost_str = f"${option['monthly_cost']:,.2f}"
                cost_savings_str = f"${option['cost_savings_monthly']:,.2f} ({option['cost_savings_percent']:.1f}%)"
                performance_str = f"{option['performance_improvement_percent']:+.1f}%"
            
            report += f"""
{i+1}. {option['target_spec']} {savings_indicator}{performance_indicator}
   - Recommended GPUs: {option['optimal_gpu_count']}
   - Monthly Cost: {monthly_cost_str}
   - Cost Savings: {cost_savings_str}
   - Performance Change: {performance_str}
   - Total Memory: {option['total_memory_gb']}GB
   - Recommendation: {option['recommendation']}
   - Risk Level: {option['risk_level']}
"""
        
        # Add specific guidance
        report += f"""

MIGRATION GUIDANCE:
"""
        
        if workload_type == WorkloadType.TRAINING or workload_type == WorkloadType.FINE_TUNING:
            if model_size == ModelSize.LARGE or model_size == ModelSize.MASSIVE:
                report += "- Large model training requires P-series instances with NVSwitch\n"
                report += "- Consider P6 (B200) for 2x training speed improvement\n"
                report += "- P5e (H200) offers more memory for larger batch sizes\n"
            else:
                report += "- Smaller models can benefit from G6 cost optimization\n"
                report += "- Consider multi-GPU G6e if you need some interconnect capability\n"
        
        else:  # Inference
            report += "- For high-volume inference, newer P-series offer better cost per token\n"
            report += "- For moderate load, G6 (L4) provides excellent cost efficiency\n"
            report += "- Consider memory requirements for batch inference optimization\n"
        
        return report
    
    def run_workload_by_id(self, workload_id: str) -> str:
        """Run a specific workload by its ID from the configuration"""
        if workload_id not in self.workloads_config["workloads"]:
            raise ValueError(f"Workload '{workload_id}' not found in configuration")
        
        workload = self.workloads_config["workloads"][workload_id]
        scenario = workload["scenario"]
        
        # Convert string enums to actual enum values
        workload_type = WorkloadType(scenario["workload_type"].lower())
        model_size = ModelSize(scenario["model_size"].lower())
        
        print(f"WORKLOAD: {workload['name']}")
        print("=" * (len(workload['name']) + 9))
        print(f"Description: {workload['description']}")
        print(f"Expected: {workload['expected_outcome']}")
        print(f"Tags: {', '.join(workload['tags'])}")
        print()
        
        return self.generate_migration_report(
            current_instance=scenario["current_instance"],
            current_gpu_count=scenario["current_gpu_count"],
            workload_type=workload_type,
            model_size=model_size,
            monthly_hours=scenario["monthly_hours"],
            use_spot=scenario["use_spot"]
        )
    
    def run_workload_set(self, set_name: str = "basic_demos") -> None:
        """Run a set of workloads defined in the configuration"""
        if set_name not in self.workloads_config["workload_sets"]:
            available_sets = list(self.workloads_config["workload_sets"].keys())
            raise ValueError(f"Workload set '{set_name}' not found. Available sets: {available_sets}")
        
        workload_ids = self.workloads_config["workload_sets"][set_name]
        
        print(f"🚀 RUNNING WORKLOAD SET: {set_name.upper()}")
        print("=" * 80)
        print(f"Workloads: {len(workload_ids)}")
        print()
        
        for i, workload_id in enumerate(workload_ids, 1):
            print(f"\n{'='*20} WORKLOAD {i}/{len(workload_ids)} {'='*20}")
            try:
                report = self.run_workload_by_id(workload_id)
                print(report)
            except Exception as e:
                print(f"❌ Error running workload '{workload_id}': {e}")
            
            if i < len(workload_ids):  # Add separator between workloads
                print("\n" + "="*80)
        
        print(f"\n✅ Completed running {len(workload_ids)} workloads from '{set_name}' set")
    
    def list_available_workloads(self) -> None:
        """List all available workloads and workload sets"""
        print("📋 AVAILABLE WORKLOADS")
        print("=" * 40)
        
        for workload_id, workload in self.workloads_config["workloads"].items():
            tags_str = ", ".join(workload["tags"])
            print(f"🔹 {workload_id}")
            print(f"   Name: {workload['name']}")
            print(f"   Tags: {tags_str}")
            print(f"   Expected: {workload['expected_outcome']}")
            print()
        
        print("\n📦 AVAILABLE WORKLOAD SETS")
        print("=" * 40)
        
        for set_name, workload_ids in self.workloads_config["workload_sets"].items():
            print(f"🔸 {set_name}")
            print(f"   Workloads: {len(workload_ids)} ({', '.join(workload_ids)})")
            print()

# Workload usage and testing
def example_usage():
    analyzer = AWSGPUMigrationAnalyzer()
    
    # Check if workloads are available
    if not analyzer.workloads_config["workloads"]:
        print("⚠️  No workloads configuration found. Using basic demo...")
        # Fallback to hardcoded examples
        print("WORKLOAD 1: Fine-tuning 30B model on 2x P5 H100")
        print("=" * 50)
        report1 = analyzer.generate_migration_report(
            current_instance="P5",
            current_gpu_count=2, 
            workload_type=WorkloadType.FINE_TUNING,
            model_size=ModelSize.MEDIUM,
            monthly_hours=200,
            use_spot=True
        )
        print(report1)
        
        print("\n\nWORKLOAD 2: Serving 13B model on 1x P5 H100")
        print("=" * 50)
        report2 = analyzer.generate_migration_report(
            current_instance="P5",
            current_gpu_count=1,
            workload_type=WorkloadType.INFERENCE, 
            model_size=ModelSize.SMALL,
            monthly_hours=720,
            use_spot=False
        )
        print(report2)
    else:
        # Use configured workloads
        analyzer.run_workload_set("basic_demos")

if __name__ == "__main__":
    example_usage()
