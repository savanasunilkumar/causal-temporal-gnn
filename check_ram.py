#!/usr/bin/env python3
"""Check RAM availability before running benchmark."""

import subprocess
import sys

def get_memory_info():
    """Get memory info on macOS."""
    try:
        # Get total memory
        total_mem = subprocess.check_output(['sysctl', 'hw.memsize']).decode().strip()
        total_gb = int(total_mem.split(': ')[1]) / (1024**3)
        
        # Get memory pressure (macOS specific)
        vm_stat = subprocess.check_output(['vm_stat']).decode()
        
        # Parse vm_stat output
        lines = vm_stat.split('\n')
        stats = {}
        for line in lines[1:]:
            if ':' in line:
                key, value = line.split(':')
                stats[key.strip()] = int(value.strip().replace('.', ''))
        
        # Page size (usually 4096 bytes on macOS)
        page_size = 4096
        
        # Calculate memory usage
        pages_free = stats.get('Pages free', 0)
        pages_inactive = stats.get('Pages inactive', 0)
        pages_active = stats.get('Pages active', 0)
        pages_wired = stats.get('Pages wired down', 0)
        
        free_gb = (pages_free * page_size) / (1024**3)
        inactive_gb = (pages_inactive * page_size) / (1024**3)
        active_gb = (pages_active * page_size) / (1024**3)
        wired_gb = (pages_wired * page_size) / (1024**3)
        
        available_gb = free_gb + inactive_gb
        used_gb = active_gb + wired_gb
        
        return {
            'total': total_gb,
            'available': available_gb,
            'used': used_gb,
            'free': free_gb,
            'inactive': inactive_gb,
            'active': active_gb,
            'wired': wired_gb
        }
    except Exception as e:
        print(f"Error getting memory info: {e}")
        return None

def main():
    print("=" * 80)
    print(" " * 25 + "M3 RAM CHECK")
    print("=" * 80)
    print()
    
    mem = get_memory_info()
    
    if mem:
        print(f"Total RAM:      {mem['total']:.2f} GB")
        print(f"Used RAM:       {mem['used']:.2f} GB ({mem['used']/mem['total']*100:.1f}%)")
        print(f"Available RAM:  {mem['available']:.2f} GB ({mem['available']/mem['total']*100:.1f}%)")
        print()
        print("Details:")
        print(f"  - Free:       {mem['free']:.2f} GB")
        print(f"  - Inactive:   {mem['inactive']:.2f} GB (can be freed)")
        print(f"  - Active:     {mem['active']:.2f} GB")
        print(f"  - Wired:      {mem['wired']:.2f} GB")
        print()
        print("=" * 80)
        print()
        
        # Benchmark requirements
        benchmark_needs = 2.0  # GB
        
        print(f"Benchmark Requirements: ~{benchmark_needs:.1f} GB")
        print()
        
        if mem['available'] >= benchmark_needs:
            print("✅ STATUS: GOOD TO GO!")
            print(f"   You have {mem['available']:.2f} GB available")
            print(f"   Benchmark needs ~{benchmark_needs:.1f} GB")
            print(f"   Safety margin: {mem['available'] - benchmark_needs:.2f} GB")
            print()
            print("👉 Run: ./setup_and_run.sh")
            sys.exit(0)
        elif mem['available'] >= benchmark_needs * 0.8:
            print("⚠️  STATUS: TIGHT BUT SHOULD WORK")
            print(f"   You have {mem['available']:.2f} GB available")
            print(f"   Benchmark needs ~{benchmark_needs:.1f} GB")
            print()
            print("💡 RECOMMENDATION:")
            print("   - Close some apps to free up more memory")
            print("   - Chrome, Slack, etc. can use a lot of RAM")
            print("   - Or just try running it (should work)")
            print()
            print("👉 Run: ./setup_and_run.sh")
            sys.exit(0)
        else:
            print("❌ STATUS: NOT ENOUGH RAM")
            print(f"   You have {mem['available']:.2f} GB available")
            print(f"   Benchmark needs ~{benchmark_needs:.1f} GB")
            print(f"   Shortfall: {benchmark_needs - mem['available']:.2f} GB")
            print()
            print("💡 WHAT TO DO:")
            print("   1. Close heavy apps (browsers, IDEs, etc.)")
            print("   2. Restart your Mac (clears caches)")
            print("   3. Run this script again to recheck")
            print()
            sys.exit(1)
    else:
        print("❌ Could not check memory. Run manually:")
        print("   vm_stat")
        print("   sysctl hw.memsize")
    
    print("=" * 80)

if __name__ == '__main__':
    main()
