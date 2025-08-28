#!/usr/bin/env python3
"""
Configuration viewer utility for GPU Fleet Optimizer
"""

import json
import os
from tabulate import tabulate

def load_config(config_file):
    """Load configuration from JSON file"""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Configuration file not found: {config_file}")
        return None
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in configuration file: {e}")
        return None

def show_gpu_instances():
    """Display GPU instances configuration in table format"""
    config = load_config("config/gpu_instances.json")
    if not config:
        return
    
    print("\n🖥️  GPU INSTANCES CONFIGURATION")
    print("=" * 80)
    
    headers = ["Instance", "GPU Type", "Memory (GB)", "GPUs/Instance", "TFLOPS", "On-Demand $/hr", "Spot $/hr"]
    table_data = []
    
    for instance_id, instance_config in config["gpu_instances"].items():
        table_data.append([
            instance_id,
            instance_config["gpu_type"],
            instance_config["gpu_memory_gb"],
            instance_config["gpus_per_instance"],
            f"{instance_config['compute_tflops']:.1f}",
            f"${instance_config['on_demand_cost_per_hour']:.2f}",
            f"${instance_config['spot_cost_per_hour']:.2f}"
        ])
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Show detailed descriptions
    print("\n📋 INSTANCE DESCRIPTIONS:")
    for instance_id, instance_config in config["gpu_instances"].items():
        print(f"• {instance_id}: {instance_config.get('description', 'No description available')}")

def show_model_configurations():
    """Display model configurations"""
    config = load_config("config/model_config.json")
    if not config:
        return
    
    print("\n🤖 MODEL SIZE CONFIGURATIONS")
    print("=" * 80)
    
    headers = ["Size", "Parameters (B)", "Memory Overhead", "Examples"]
    table_data = []
    
    for size_key, size_config in config["model_sizes"].items():
        examples = ", ".join(size_config["examples"][:2])  # Show first 2 examples
        if len(size_config["examples"]) > 2:
            examples += "..."
        
        table_data.append([
            size_config["name"],
            size_config["parameter_count_billion"],
            f"{size_config['memory_overhead_multiplier']}x",
            examples
        ])
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

def show_precision_formats():
    """Display precision formats configuration"""
    config = load_config("config/model_config.json")
    if not config:
        return
    
    print("\n🎯 PRECISION FORMATS")
    print("=" * 60)
    
    headers = ["Format", "Bytes/Param", "Performance Impact", "Description"]
    table_data = []
    
    for format_key, format_config in config["precision_formats"].items():
        table_data.append([
            format_config["name"],
            format_config["bytes_per_parameter"],
            f"{format_config['performance_impact']*100:.0f}%",
            format_config["description"]
        ])
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

def show_workload_types():
    """Display workload types configuration"""
    config = load_config("config/model_config.json")
    if not config:
        return
    
    print("\n⚡ WORKLOAD TYPES")
    print("=" * 60)
    
    for workload_key, workload_config in config["workload_types"].items():
        print(f"\n🔹 {workload_config['name']}")
        print(f"   Description: {workload_config['description']}")
        print(f"   Memory Requirements: {workload_config['memory_requirements']}")
        print(f"   Compute Intensity: {workload_config['compute_intensity']}")
        print(f"   Typical Duration: {workload_config['typical_duration']}")
        print(f"   Recommended Families: {', '.join(workload_config['recommended_instance_families'])}")

def show_use_cases():
    """Display use cases configuration"""
    config = load_config("config/model_config.json")
    if not config:
        return
    
    print("\n🎯 USE CASES")
    print("=" * 60)
    
    headers = ["Use Case", "Workload", "Priority", "Cost Sensitivity", "Spot Suitable"]
    table_data = []
    
    for use_case_key, use_case_config in config["use_cases"].items():
        table_data.append([
            use_case_config["name"],
            use_case_config["workload_type"],
            use_case_config["priority"],
            use_case_config["cost_sensitivity"],
            "Yes" if use_case_config["spot_instance_suitable"] else "No"
        ])
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

def main():
    """Main function to display all configurations"""
    print("🚀 GPU FLEET OPTIMIZER - CONFIGURATION VIEWER")
    print("=" * 80)
    
    try:
        show_gpu_instances()
        show_model_configurations()
        show_precision_formats()
        show_workload_types()
        show_use_cases()
        
        print("\n✅ Configuration files loaded successfully!")
        print("📝 To modify configurations, edit files in the config/ directory")
        print("🔄 Changes will take effect on next analyzer initialization")
        
    except Exception as e:
        print(f"❌ Error loading configurations: {e}")

if __name__ == "__main__":
    main()
