#!/usr/bin/env python3
"""
Quick script to add random IP addresses to existing network_topology.json
Creates network_topology2.json with same positioning but added IP info
"""

import json
import random
import ipaddress
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

def generate_random_port():
    """Generate random port numbers (common ranges)"""
    common_ports = [80, 443, 22, 21, 25, 53, 110, 993, 995, 3389, 5432, 3306]
    if random.random() < 0.3:  # 30% chance for common ports
        return random.choice(common_ports)
    else:
        return random.randint(1024, 65535)  # Random high ports

def add_ips_to_topology():
    """Add random IP addresses to network topology"""
    
    # Paths
    input_file = Path("Data/network_topology.json")
    output_file = Path("Data/network_topology2.json")
    
    print(f"🔄 Loading {input_file}")
    
    # Load existing topology
    with open(input_file, 'r') as f:
        topology = json.load(f)
    
    print(f"📊 Found {len(topology['nodes'])} nodes to process")
    
    # Update metadata
    topology['metadata']['dataset_name'] = "CIC IDS Dataset (Sample) - With Random IPs"
    topology['metadata']['network_info_included'] = True
    topology['metadata']['ip_generation'] = "random_for_proof_of_concept"
    
    # Add IP information to each node
    used_source_ips = set()
    protocols = ["TCP", "UDP", "ICMP"]
    
    for i, node in enumerate(topology['nodes']):
        # Generate unique source IP for this node
        while True:
            source_ip = generate_random_ip()
            if source_ip not in used_source_ips:
                used_source_ips.add(source_ip)
                break
        
        # Generate destination IP (can be duplicate)
        destination_ip = generate_random_ip()
        
        # Generate ports and protocol
        source_port = generate_random_port()
        destination_port = generate_random_port()
        protocol = random.choice(protocols)
        
        # Add network information
        node["source_ip"] = source_ip
        node["destination_ip"] = destination_ip
        node["source_port"] = source_port
        node["destination_port"] = destination_port
        node["protocol"] = protocol
        
        # Add detailed network info section
        node["network_info"] = {
            "src_ip": source_ip,
            "dst_ip": destination_ip,
            "src_port": source_port,
            "dst_port": destination_port,
            "protocol": protocol,
            "connection_type": "simulated",
            "data_source": "random_generation"
        }
        
        # Update label to include IP
        node["label"] = f"Device_{i} ({source_ip})"
        
        if (i + 1) % 1000 == 0:
            print(f"  ✓ Processed {i + 1} nodes...")
    
    print(f"🌐 Generated {len(used_source_ips)} unique source IPs")
    print(f"🔗 Processed {len(topology['edges'])} edges")
    
    # Update statistics
    topology['metadata']['unique_source_ips'] = len(used_source_ips)
    topology['metadata']['ip_ranges_used'] = [
        "192.168.x.x (Home networks)",
        "10.0.x.x (Corporate networks)", 
        "172.16.x.x (Private networks)"
    ]
    
    # Save new topology
    print(f"💾 Saving to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(topology, f, indent=2)
    
    print(f"✅ Created {output_file}")
    print(f"📈 Summary:")
    print(f"   • Nodes: {len(topology['nodes'])}")
    print(f"   • Edges: {len(topology['edges'])}")
    print(f"   • Unique Source IPs: {len(used_source_ips)}")
    print(f"   • Positioning: Identical to original")
    print(f"   • Ready for UE5 import!")

if __name__ == "__main__":
    add_ips_to_topology()
