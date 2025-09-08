#!/usr/bin/env python3
"""
Simple script to add only IP addresses to existing network_topology.json
Creates network_topology2.json with same structure but added IP field
"""

import json
import random
from pathlib import Path

def generate_random_ip():
    """Generate a random IP address from common private ranges"""
    ranges = [
        "192.168.{}.{}",  # Home networks
        "10.0.{}.{}",     # Corporate networks  
        "172.16.{}.{}"    # Private networks
    ]
    
    range_template = random.choice(ranges)
    return range_template.format(
        random.randint(1, 254),
        random.randint(1, 254)
    )

def add_simple_ips():
    """Add only IP addresses to network topology"""
    
    # Paths
    input_file = Path("Data/network_topology.json")
    output_file = Path("Data/network_topology2.json")
    
    print(f"🔄 Loading {input_file}")
    
    # Load existing topology
    with open(input_file, 'r') as f:
        topology = json.load(f)
    
    print(f"📊 Found {len(topology['nodes'])} nodes to process")
    
    # Generate unique IP addresses
    used_ips = set()
    
    # Add IP address to each node
    for i, node in enumerate(topology['nodes']):
        # Generate unique IP for this node
        while True:
            ip_address = generate_random_ip()
            if ip_address not in used_ips:
                used_ips.add(ip_address)
                break
        
        # Add only IP address field
        node["ip_address"] = ip_address
        
        if (i + 1) % 25 == 0:
            print(f"  ✓ Processed {i + 1} nodes...")
    
    print(f"🌐 Generated {len(used_ips)} unique IP addresses")
    
    # Save new topology
    print(f"💾 Saving to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(topology, f, indent=2)
    
    print(f"✅ Created {output_file}")
    print(f"📈 Summary:")
    print(f"   • Nodes: {len(topology['nodes'])}")
    print(f"   • Edges: {len(topology['edges'])}")
    print(f"   • Unique IPs: {len(used_ips)}")
    print(f"   • Added field: 'ip_address' only")
    print(f"   • Positioning: Identical to original")

if __name__ == "__main__":
    add_simple_ips()
