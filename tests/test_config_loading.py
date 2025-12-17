#!/usr/bin/env python3
"""
Test script to verify configuration loading from config.yml
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_config_loading():
    """Test that configuration loads correctly from config.yml"""
    
    # Set the config path to use our config.yml
    os.environ['RERANKER_CONFIG_PATH'] = str(Path(__file__).parent / 'config.yml')
    
    # Import settings (will load config)
    from src.config import settings
    
    print("=" * 60)
    print("Configuration Test Results")
    print("=" * 60)
    
    # Test server config
    print("\n[Server Configuration]")
    print(f"  Host: {settings.host}")
    print(f"  Port: {settings.port}")
    print(f"  Workers: {settings.workers}")
    print(f"  Reload: {settings.reload}")
    
    # Test model config
    print("\n[Model Configuration]")
    print(f"  Model Name: {settings.model_name}")
    print(f"  Model Path: {settings.model_path}")
    print(f"  Cache Dir: {settings.model_cache_dir}")
    print(f"  Offline Mode: {settings.use_offline_mode}")
    
    # Test inference config
    print("\n[Inference Configuration]")
    print(f"  Max Length: {settings.max_length}")
    print(f"  Batch Size: {settings.batch_size}")
    print(f"  Normalize Scores: {settings.normalize_scores}")
    
    # Test device config
    print("\n[Device Configuration]")
    print(f"  Device: {settings.device}")
    print(f"  Force CPU Only: {settings.force_cpu_only}")
    print(f"  Use FP16: {settings.use_fp16}")
    print(f"  MPS Fallback: {settings.mps_fallback_to_cpu}")
    print(f"  Auto-detected Device: {settings.get_device()}")
    
    # Test API config
    print("\n[API Configuration]")
    print(f"  API Key: {'***' if settings.api_key else 'None'}")
    print(f"  Enable CORS: {settings.enable_cors}")
    print(f"  CORS Origins: {settings.cors_origins}")
    
    # Test async engine config
    print("\n[Async Engine Configuration]")
    print(f"  Enabled: {settings.enable_async_engine}")
    print(f"  Max Concurrent Batches: {settings.max_concurrent_batches}")
    print(f"  Inference Threads: {settings.inference_threads}")
    print(f"  Max Batch Size: {settings.max_batch_size}")
    print(f"  Max Batch Pairs: {settings.max_batch_pairs}")
    print(f"  Batch Wait Timeout: {settings.batch_wait_timeout}s")
    print(f"  Max Queue Size: {settings.max_queue_size}")
    print(f"  Request Timeout: {settings.request_timeout}s")
    
    # Test load balancer config
    print("\n[Load Balancer Configuration]")
    print(f"  Enabled: {settings.enable_load_balancer}")
    print(f"  Config Path: {settings.config_path}")
    
    # Test HTTP config
    print("\n[HTTP Configuration]")
    print(f"  Trust Env: {settings.trust_env}")
    
    # Test logging config
    print("\n[Logging Configuration]")
    print(f"  Log Level: {settings.log_level}")
    print(f"  JSON Logs: {settings.json_logs}")
    print(f"  Log Dir: {settings.log_dir}")
    print(f"  Retention Days: {settings.log_retention_days}")
    print(f"  Max Bytes: {settings.log_max_bytes}")
    print(f"  Backup Count: {settings.log_backup_count}")
    
    print("\n" + "=" * 60)
    print("✓ Configuration loaded successfully from config.yml!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_config_loading()
    except Exception as e:
        print(f"\n❌ Error loading configuration: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
