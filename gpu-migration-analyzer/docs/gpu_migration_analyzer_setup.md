# GPU Fleet Optimizer - Complete Setup & Configuration Guide

A comprehensive guide to setting up, configuring, and using the GPU Fleet Optimizer with its flexible JSON-based configuration system.


## 📁 Configuration System

### 1. GPU Instances Configuration (`config/gpu_instances.json`)

Defines all available GPU instance types with specifications and pricing:

```json
{
  "gpu_instances": {
    "P5": {
      "name": "P5 (H100)",
      "family": "P",
      "gpu_type": "NVIDIA H100",
      "vcpus": 192,
      "gpu_memory_gb": 80,
      "gpus_per_instance": 8,
      "compute_tflops": 2000.0,
      "interconnect_type": "NVSwitch",
      "interconnect_bandwidth_gbps": 900,
      "on_demand_cost_per_hour": 50.0,
      "spot_cost_per_hour": 16.0,
      "memory_bandwidth_factor": 0.9,
      "interconnect_multiplier": 1.0,
      "description": "Latest generation high-performance training instance"
    },
    "G6e": {
      "name": "G6e (L40S)",
      "family": "G",
      "gpu_type": "NVIDIA L40S",
      "vcpus": 192,
      "gpu_memory_gb": 48,
      "gpus_per_instance": 8,
      "compute_tflops": 733.0,
      "interconnect_type": "NVLink 4-way",
      "interconnect_bandwidth_gbps": 450,
      "on_demand_cost_per_hour": 3.5,
      "spot_cost_per_hour": 1.75,
      "memory_bandwidth_factor": 0.6,
      "interconnect_multiplier": 0.7,
      "description": "High-performance GPU instance with enhanced interconnect"
    }
  }
}
```

**Available Instances**: G4dn, G5, G6, G6e, P4d, P4de, P5, P5e, P6

### 2. Model & Workload Configuration (`config/model_config.json`)

#### Model Sizes
```json
{
  "model_sizes": {
    "small": {
      "name": "Small",
      "description": "1-13B parameters",
      "parameter_count_billion": 13,
      "typical_use_cases": ["inference", "fine_tuning", "experimentation"],
      "memory_overhead_multiplier": 1.5,
      "examples": ["Llama 2 7B", "Mistral 7B", "CodeLlama 7B"]
    },
    "medium": {
      "name": "Medium", 
      "description": "14-70B parameters",
      "parameter_count_billion": 35,
      "memory_overhead_multiplier": 1.5,
      "examples": ["Llama 2 70B", "CodeLlama 34B", "Mixtral 8x7B"]
    }
  }
}
```

#### Precision Formats (Memory Usage)
- **FP32**: 4 bytes/parameter (highest accuracy, largest memory)
- **FP16**: 2 bytes/parameter (balanced accuracy/memory)
- **BF16**: 2 bytes/parameter (better numerical stability)
- **FP8**: 1 byte/parameter (experimental, inference optimization)
- **INT8**: 1 byte/parameter (quantized, lowest memory)

### 3. Workload Scenarios (`config/workloads.json`)

Pre-defined realistic migration scenarios with expected outcomes:

```json
{
  "workloads": {
    "inference_small_p5": {
      "name": "Serving 13B model on 1x P5 H100",
      "description": "Customer doing inference on over-provisioned instance",
      "scenario": {
        "current_instance": "P5",
        "current_gpu_count": 1,
        "workload_type": "INFERENCE",
        "model_size": "SMALL",
        "monthly_hours": 720,
        "use_spot": false,
        "precision": "fp16"
      },
      "expected_outcome": "Significant cost savings with G6/G6e instances",
      "tags": ["inference", "small_model", "over_provisioned", "cost_reduction"]
    }
  },
  "workload_sets": {
    "basic_demos": ["fine_tuning_medium_p5", "inference_small_p5"],
    "cost_optimization_focus": ["inference_small_p5", "batch_inference_g5", "production_serving_g4dn"],
    "performance_focus": ["training_large_p4d", "inference_massive_p5e"],
    "comprehensive_suite": ["fine_tuning_medium_p5", "inference_small_p5", "training_large_p4d", "inference_massive_p5e"],
    "research_academic": ["research_training_g6e", "batch_inference_g5"]
  }
}
```

## 🎯 How to Run the Analyzer

### Method 1: Default Workloads
```bash
python3 gpu_migration_analyzer.py
```
Runs the `basic_demos` workload set automatically.

### Method 2: Specific Workload
```python
python3 -c "
from gpu_migration_analyzer import AWSGPUMigrationAnalyzer
analyzer = AWSGPUMigrationAnalyzer()
analyzer.run_workload_by_id('inference_small_p5')
"
```

### Method 3: Workload Set (Themed Collections)
```python
python3 -c "
from gpu_migration_analyzer import AWSGPUMigrationAnalyzer
analyzer = AWSGPUMigrationAnalyzer()
analyzer.run_workload_set('cost_optimization_focus')
"
```

### Method 4: Custom Analysis
```python
python3 -c "
from gpu_migration_analyzer import AWSGPUMigrationAnalyzer, WorkloadType, ModelSize
analyzer = AWSGPUMigrationAnalyzer()
report = analyzer.generate_migration_report(
    current_instance='P5',
    current_gpu_count=2,
    workload_type=WorkloadType.TRAINING,
    model_size=ModelSize.LARGE,
    monthly_hours=500,
    use_spot=True
)
print(report)
"
```

### Method 5: Interactive Python Session
```python
from gpu_migration_analyzer import AWSGPUMigrationAnalyzer, WorkloadType, ModelSize

# Initialize analyzer
analyzer = AWSGPUMigrationAnalyzer()

# List available workloads
analyzer.list_available_workloads()

# Run specific workload
analyzer.run_workload_by_id('training_large_p4d')

# Custom analysis with different precision
report = analyzer.generate_migration_report(
    current_instance="P5", current_gpu_count=1,
    workload_type=WorkloadType.INFERENCE, model_size=ModelSize.MEDIUM,
    precision="int8"  # Uses 35GB instead of 70GB for FP16
)
print(report)
```

## 💰 Cost Comparison & Decision Making

### Instance Cost Comparison (Per Hour)
| Instance | GPU Type | Memory (GB) | On-Demand $/hr | Spot $/hr | Monthly Cost (On-Demand/Spot) |
|----------|----------|-------------|-----------------|-----------|-------------------------------|
| G4dn     | T4       | 16          | $0.53           | $0.21     | $381 / $154                   |
| G5       | A10G     | 24          | $1.62           | $0.44     | $1,166 / $317                 |
| G6       | L4       | 24          | $1.32           | $0.73     | $950 / $525                   |
| G6e      | L40S     | 48          | $3.50           | $1.75     | $2,520 / $1,260              |
| P4d      | A100 40GB| 40          | $32.00          | $10.00    | $23,040 / $7,200             |
| P4de     | A100 80GB| 80          | $40.00          | $12.00    | $28,800 / $8,640             |
| P5       | H100     | 80          | $50.00          | $16.00    | $36,000 / $11,520            |
| P5e      | H200     | 141         | $55.00          | $18.00    | $39,600 / $12,960            |
| P6       | B200     | 192         | $60.00          | $20.00    | $43,200 / $14,400            |

*Monthly costs calculated for 1 GPU @ 720 hours*

## 🛠️ Advanced Usage Examples

### 1. Precision Format Comparison
```python
analyzer = AWSGPUMigrationAnalyzer()

# Compare memory usage across precision formats for Medium model (35B params)
precisions = ["fp32", "fp16", "bf16", "fp8", "int8"]
for precision in precisions:
    memory_gb = analyzer.get_model_size_gb(ModelSize.MEDIUM, precision)
    print(f"{precision.upper()}: {memory_gb}GB")

# Output:
# FP32: 140GB
# FP16: 70GB  
# BF16: 70GB
# FP8: 35GB
# INT8: 35GB
```

### 2. Batch Analysis for Multiple Current Setups
```python
scenarios = [
    {"instance": "G4dn", "gpus": 4, "workload": WorkloadType.INFERENCE, "model": ModelSize.SMALL},
    {"instance": "P4d", "gpus": 8, "workload": WorkloadType.TRAINING, "model": ModelSize.LARGE},
    {"instance": "G6e", "gpus": 2, "workload": WorkloadType.FINE_TUNING, "model": ModelSize.MEDIUM}
]

for scenario in scenarios:
    print(f"\n=== {scenario['instance']} Analysis ===")
    report = analyzer.generate_migration_report(
        current_instance=scenario["instance"],
        current_gpu_count=scenario["gpus"],
        workload_type=scenario["workload"],
        model_size=scenario["model"],
        monthly_hours=720,
        use_spot=True
    )
    print(report)
```

### 3. Spot vs On-Demand Cost Analysis
```python
# Compare costs for the same workload
instances = ["P5", "G6e", "P4d"]
for instance in instances:
    print(f"\n=== {instance} Cost Comparison ===")
    
    # On-demand pricing
    report_ondemand = analyzer.generate_migration_report(
        current_instance=instance, current_gpu_count=2,
        workload_type=WorkloadType.TRAINING, model_size=ModelSize.LARGE,
        use_spot=False
    )
    
    # Spot pricing
    report_spot = analyzer.generate_migration_report(
        current_instance=instance, current_gpu_count=2,
        workload_type=WorkloadType.TRAINING, model_size=ModelSize.LARGE,
        use_spot=True
    )
```

## 📦 Workload Sets - Choose by Your Goal

### Available Workload Sets:
1. **`basic_demos`** - Start here: Basic migration scenarios
2. **`comprehensive_suite`** - Complete analysis: All workload types  
3. **`cost_optimization_focus`** - Save money: Cost reduction scenarios
4. **`performance_focus`** - Go faster: Performance upgrade scenarios
5. **`research_academic`** - Budget-conscious: Academic & research scenarios

### Running Workload Sets:
```python
# Cost optimization focus
analyzer.run_workload_set('cost_optimization_focus')

# Performance upgrade focus  
analyzer.run_workload_set('performance_focus')

# Academic research scenarios
analyzer.run_workload_set('research_academic')
```

## 🔧 Customization & Configuration

### 1. Update Pricing
Edit `config/gpu_instances.json`:
```json
{
  "P5": {
    "on_demand_cost_per_hour": 45.0,  // Updated price
    "spot_cost_per_hour": 15.0        // Updated spot price
  }
}
```

### 2. Add New GPU Instance
```json
{
  "gpu_instances": {
    "NewGPU": {
      "name": "New Instance (Custom GPU)",
      "family": "N",
      "gpu_type": "NVIDIA CustomGPU",
      "vcpus": 128,
      "gpu_memory_gb": 96,
      "gpus_per_instance": 4,
      "compute_tflops": 1500.0,
      "interconnect_type": "Custom",
      "interconnect_bandwidth_gbps": 800,
      "on_demand_cost_per_hour": 45.0,
      "spot_cost_per_hour": 15.0,
      "memory_bandwidth_factor": 0.85,
      "interconnect_multiplier": 0.9,
      "description": "Custom GPU instance for specific workloads"
    }
  }
}
```

### 3. Create Custom Workload Scenario
Edit `config/workloads.json`:
```json
{
  "workloads": {
    "my_custom_workload": {
      "name": "Custom Training Scenario",
      "description": "Large scale model training on legacy hardware",
      "scenario": {
        "current_instance": "P4d",
        "current_gpu_count": 8,
        "workload_type": "TRAINING",
        "model_size": "MASSIVE", 
        "monthly_hours": 300,
        "use_spot": true,
        "precision": "bf16"
      },
      "expected_outcome": "Significant performance gains with newer P-series",
      "tags": ["training", "massive_model", "performance_upgrade"]
    }
  }
}
```

### 4. Add Custom Model Size
Edit `config/model_config.json`:
```json
{
  "model_sizes": {
    "extra_large": {
      "name": "Extra Large",
      "description": "200B+ parameters",
      "parameter_count_billion": 200,
      "typical_use_cases": ["research", "enterprise_applications"],
      "memory_overhead_multiplier": 1.8,
      "examples": ["GPT-4+", "Custom Enterprise Models"]
    }
  }
}
```

## 💡 Best Practices & Tips

### Workload Selection Guidelines:
- **Training**: Use P-series instances with NVSwitch for large models
- **Inference**: G-series often provides better cost efficiency  
- **Fine-tuning**: Consider G6e as cost-effective alternative to P-series
- **Spot Instances**: Use for training (60-70% cost savings), avoid for production inference

### Memory Planning:
- **Training overhead**: Model size × 1.5 (for optimizer states, gradients)
- **Inference**: Model size × 1.2 (for batching and context)
- **Precision impact**: FP16 uses 50% memory vs FP32, INT8 uses 25%

### Common Migration Patterns:
1. **Over-provisioned inference**: P5/P4d → G6e/G6 (90%+ cost savings)
2. **Legacy training**: P4d → P5/P6 (2-3x performance improvement)
3. **Budget research**: Any instance → spot pricing (60-70% savings)

## 🚨 Common Issues & Solutions

### "CANNOT FIT MODEL" Error
**Cause**: Model too large for GPU memory
**Solutions**: 
- Use INT8/FP8 precision to reduce memory usage
- Increase GPU count
- Choose instance with more memory per GPU (P5e, P6)

### High Monthly Costs
**Cause**: Using on-demand pricing or over-provisioned instances
**Solutions**:
- Enable spot pricing (`use_spot=True`)
- Right-size instance for workload
- Consider G-series for inference workloads

### Low Performance Recommendations
**Cause**: Current setup already well-optimized
**Solutions**:
- Consider newer GPU generations (P6 vs P5)
- Evaluate multi-GPU configurations
- Check if workload type matches instance strengths

## 📋 Configuration Validation

The analyzer automatically validates configurations:
- **Startup validation**: Checks for missing files and invalid JSON
- **Runtime validation**: Validates enum values and required fields
- **Memory validation**: Ensures models can fit in specified GPU configurations

### Troubleshooting Configuration Issues:
```bash
# Check configuration status
python3 config_viewer.py

# Test specific workload
python3 -c "
analyzer = AWSGPUMigrationAnalyzer()
analyzer.run_workload_by_id('inference_small_p5')
"
```

## 🎯 Next Steps

1. **Start Simple**: Run `python3 gpu_migration_analyzer.py` for basic demo
2. **Explore Options**: Use `python3 config_viewer.py` to see all available configurations
3. **Run Themed Sets**: Try `cost_optimization_focus` or `performance_focus` workload sets
4. **Customize**: Modify configurations to match your specific GPU instances and pricing
5. **Integrate**: Use the API in your own scripts for automated migration analysis

---

📝 **Need Help?** 
- Run `python3 config_viewer.py` for interactive guidance
- Check workload scenarios in `config/workloads.json`
- Review this guide for setup and usage examples



### Workload Sets by Business Goal

| Set Name | Purpose | Scenarios | Use Case |
|----------|---------|-----------|----------|
| `basic_demos` | Getting Started | 2 scenarios | New users exploring the tool |
| `cost_optimization_focus` | Cost Reduction | 3 scenarios | Organizations prioritizing cost savings |
| `performance_focus` | Performance Upgrade | 2 scenarios | Teams needing better performance |
| `comprehensive_suite` | Complete Analysis | 4 scenarios | Thorough evaluation across workload types |
| `research_academic` | Budget-Conscious | 2 scenarios | Academic and research institutions |

## 💡 Key Features

### 🔧 **Migration Analyzer**
- **9 GPU Instance Types**: G4dn, G5, G6, G6e, P4d, P4de, P5, P5e, P6
- **Multiple Precision Formats**: FP32, FP16, BF16, FP8, INT8 with automatic memory calculations
- **Workload Optimization**: Training, Fine-tuning, Inference scenarios
- **Cost Analysis**: On-demand vs Spot pricing with 60-70% potential savings
- **Memory Validation**: Prevents "cannot fit model" scenarios with clear feedback
- **Professional API**: Enterprise-ready with workload-based terminology

### 📈 **Cost Analysis Features**
- Monthly cost projections with detailed breakdowns
- Spot vs on-demand pricing comparisons
- Performance improvement quantification
- Memory requirement validation
- Risk assessment for migration decisions

### 🎯 **Business-Focused Recommendations**
- Recommendations aligned with business priorities (cost, performance, reliability)
- Risk levels and migration complexity assessment
- Expected outcomes for each scenario
- Best practices and common migration patterns

## 🏗️ Repository Structure

```
gpu-fleet-optimizer/
├── README.md                           # This file
├── gpu-migration-analyzer/             # Migration analysis tool
│   ├── gpu_migration_analyzer.py       # Main analyzer
│   ├── config_viewer.py               # Configuration guide & usage helper
│   ├── config/
│   │   ├── gpu_instances.json         # GPU specifications & pricing
│   │   ├── model_config.json          # Model sizes & precision formats
│   │   └── workloads.json            # Pre-defined migration scenarios
│   └── docs/
│       └── config_viewer.md           # Complete setup & usage guide
├── spot-optimizer/                     # Coming soon
├── multi-region-analyzer/             # Coming soon
├── batch-scheduler/                    # Coming soon
└── auto-scaling-analyzer/             # Coming soon
```

## 🎓 Usage Examples

### Run Pre-configured Migration Scenarios
```bash
cd gpu-migration-analyzer

# Run basic demonstration scenarios
python3 gpu_migration_analyzer.py

# Run cost-focused scenarios
python3 -c "
from gpu_migration_analyzer import AWSGPUMigrationAnalyzer
analyzer = AWSGPUMigrationAnalyzer()
analyzer.run_workload_set('cost_optimization_focus')
"

# Run performance-focused scenarios  
python3 -c "
from gpu_migration_analyzer import AWSGPUMigrationAnalyzer
analyzer = AWSGPUMigrationAnalyzer()
analyzer.run_workload_set('performance_focus')
"
```

### Custom Migration Analysis
```python
from gpu_migration_analyzer import AWSGPUMigrationAnalyzer, WorkloadType, ModelSize

analyzer = AWSGPUMigrationAnalyzer()

# Analyze training workload migration
report = analyzer.generate_migration_report(
    current_instance='P4d',
    current_gpu_count=8,
    workload_type=WorkloadType.TRAINING,
    model_size=ModelSize.LARGE,
    monthly_hours=500,
    use_spot=True  # 60-70% cost savings
)
print(report)
```

### Compare Different Precision Formats
```python
# Compare memory usage across precision formats
precisions = ["fp32", "fp16", "bf16", "fp8", "int8"]
for precision in precisions:
    memory_gb = analyzer.get_model_size_gb(ModelSize.MEDIUM, precision)
    print(f"{precision.upper()}: {memory_gb}GB")

# Output for 35B parameter model:
# FP32: 140GB    # Highest accuracy, largest memory
# FP16: 70GB     # Balanced accuracy/memory
# BF16: 70GB     # Better numerical stability
# FP8: 35GB      # Experimental, inference optimization
# INT8: 35GB     # Quantized, lowest memory
```

## 📋 Configuration & Customization

All tools use JSON-based configuration for easy customization:

- **Update Pricing**: Modify `config/gpu_instances.json` for current market rates
- **Add Instances**: Include new GPU instance types as they become available
- **Custom Scenarios**: Create organization-specific workload scenarios
- **Model Parameters**: Adjust model sizes and precision formats

## 🚨 Common Use Cases & Solutions

| Problem | Solution | Tool | Expected Savings |
|---------|----------|------|------------------|
| High inference costs on P5 instances | Migrate to G6e for inference workloads | Migration Analyzer | 90%+ cost reduction |
| Slow training on older GPU generations | Upgrade to P5/P6 instances | Migration Analyzer | 2-3x performance improvement |
| Budget constraints for research | Enable spot pricing | Migration Analyzer | 60-70% cost savings |
| Over-provisioned legacy deployments | Right-size with modern instances | Migration Analyzer | Significant consolidation |

