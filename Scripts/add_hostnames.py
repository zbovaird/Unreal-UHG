#!/usr/bin/env python3
"""
Script to add unique hostnames (6-7 characters) to each node in network_topology2.json
Creates realistic-looking hostnames based on device types and attack patterns
"""

import json
import random
from pathlib import Path

def generate_hostname_pools():
    """Generate pools of hostname components for realistic names"""
    
    # Prefixes for different device types
    prefixes = {
        'server': ['srv', 'web', 'db', 'app', 'api', 'mail', 'dns', 'ftp'],
        'workstation': ['ws', 'pc', 'dev', 'usr', 'wks', 'desk', 'lap'],
        'network': ['rtr', 'sw', 'fw', 'gw', 'lb', 'vpn', 'wifi'],
        'iot': ['cam', 'sen', 'iot', 'hub', 'ctrl', 'mon'],
        'mobile': ['mob', 'tab', 'phn', 'byod']
    }
    
    # Suffixes for locations/departments  
    suffixes = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                'x', 'y', 'z', 'ny', 'sf', 'la', 'chi', 'atl', 'sea']
    
    # Additional middle components
    middle = ['', 'pr', 'ts', 'bk', 'dm', 'mg', 'hr', 'it', 'fn', 'op']
    
    return prefixes, suffixes, middle

def generate_unique_hostname(used_hostnames, attack_type=None):
    """Generate a unique hostname 6-7 characters long"""
    
    prefixes, suffixes, middle = generate_hostname_pools()
    
    # Weight device types based on attack type for realism
    if attack_type and attack_type != "BENIGN":
        # Malicious traffic more likely from workstations/mobile
        device_weights = {
            'workstation': 0.4,
            'server': 0.2, 
            'network': 0.1,
            'iot': 0.2,
            'mobile': 0.1
        }
    else:
        # Benign traffic more evenly distributed
        device_weights = {
            'server': 0.3,
            'workstation': 0.3,
            'network': 0.2,
            'iot': 0.1,
            'mobile': 0.1
        }
    
    max_attempts = 1000
    for _ in range(max_attempts):
        # Choose device type based on weights
        device_type = random.choices(
            list(device_weights.keys()),
            weights=list(device_weights.values())
        )[0]
        
        prefix = random.choice(prefixes[device_type])
        suffix = random.choice(suffixes)
        
        # Sometimes add middle component for variety
        if len(prefix) <= 3 and random.random() < 0.3:
            mid = random.choice(middle)
            if mid:  # Skip empty string
                hostname = f"{prefix}{mid}{suffix}"
            else:
                hostname = f"{prefix}{suffix}"
        else:
            hostname = f"{prefix}{suffix}"
        
        # Ensure 6-7 characters
        if 6 <= len(hostname) <= 7 and hostname not in used_hostnames:
            return hostname
    
    # Fallback: generate simple alphanumeric
    for i in range(max_attempts):
        hostname = f"host{random.randint(10, 99)}{random.choice('abcdefghijk')}"
        if hostname not in used_hostnames:
            return hostname
    
    # Final fallback
    return f"node{random.randint(100, 999)}"[:7]

def add_hostnames_to_topology():
    """Add unique hostnames to network topology"""
    
    # Path to the file
    file_path = Path("Data/network_topology2.json")
    
    print(f"🔄 Loading {file_path}")
    
    # Load topology
    with open(file_path, 'r') as f:
        topology = json.load(f)
    
    nodes = topology['nodes']
    total_nodes = len(nodes)
    
    print(f"🏷️  Generating unique hostnames for {total_nodes} nodes")
    
    used_hostnames = set()
    hostname_stats = {
        'total': 0,
        'by_attack_type': {},
        'length_distribution': {6: 0, 7: 0}
    }
    
    # Generate hostname for each node
    for i, node in enumerate(nodes):
        attack_type = node.get('attack_type', 'BENIGN')
        
        # Generate unique hostname
        hostname = generate_unique_hostname(used_hostnames, attack_type)
        used_hostnames.add(hostname)
        
        # Add to node
        node['hostname'] = hostname
        
        # Update label to include hostname
        node['label'] = f"{hostname} ({node.get('ip_address', 'unknown')})"
        
        # Update stats
        hostname_stats['total'] += 1
        hostname_stats['by_attack_type'][attack_type] = hostname_stats['by_attack_type'].get(attack_type, 0) + 1
        hostname_stats['length_distribution'][len(hostname)] += 1
        
        if (i + 1) % 20 == 0:
            print(f"   ✓ Generated {i + 1}/{total_nodes} hostnames")
    
    # Update metadata
    topology.setdefault('metadata', {})
    topology['metadata']['hostname_info'] = {
        'total_hostnames': len(used_hostnames),
        'hostname_length_range': "6-7 characters",
        'generation_method': "Device-type aware with attack pattern weighting",
        'uniqueness': "100% unique within topology"
    }
    
    # Sample hostnames for reference
    sample_hostnames = list(used_hostnames)[:10]
    topology['metadata']['sample_hostnames'] = sample_hostnames
    
    print(f"📊 Hostname Statistics:")
    print(f"   • Total unique hostnames: {hostname_stats['total']}")
    print(f"   • 6-character hostnames: {hostname_stats['length_distribution'][6]}")
    print(f"   • 7-character hostnames: {hostname_stats['length_distribution'][7]}")
    print(f"   • Sample hostnames: {', '.join(sample_hostnames)}")
    
    print(f"📋 Distribution by attack type:")
    for attack_type, count in hostname_stats['by_attack_type'].items():
        print(f"   • {attack_type}: {count} hostnames")
    
    # Save updated topology
    print(f"💾 Saving updated {file_path}")
    with open(file_path, 'w') as f:
        json.dump(topology, f, indent=2)
    
    print(f"🎯 Added unique hostnames to all {total_nodes} nodes!")

if __name__ == "__main__":
    # Set random seed for reproducible results
    random.seed(42)
    add_hostnames_to_topology()
