#!/usr/bin/env python3
"""
Script to fix edge type distribution to be more realistic:
- 90% normal/benign edges
- 5% suspicious edges  
- 5% malicious edges
"""

import json
import random
from pathlib import Path

def fix_edge_distribution():
    """Update edge types to realistic distribution"""
    
    # Path to the file
    file_path = Path("Data/network_topology2.json")
    
    print(f"🔄 Loading {file_path}")
    
    # Load topology
    with open(file_path, 'r') as f:
        topology = json.load(f)
    
    edges = topology['edges']
    total_edges = len(edges)
    
    print(f"📊 Found {total_edges} edges to redistribute")
    
    # Calculate target counts
    normal_count = int(total_edges * 0.90)  # 90%
    suspicious_count = int(total_edges * 0.05)  # 5%
    malicious_count = total_edges - normal_count - suspicious_count  # Remaining ~5%
    
    print(f"🎯 Target distribution:")
    print(f"   • Normal/Benign: {normal_count} ({normal_count/total_edges*100:.1f}%)")
    print(f"   • Suspicious: {suspicious_count} ({suspicious_count/total_edges*100:.1f}%)")
    print(f"   • Malicious: {malicious_count} ({malicious_count/total_edges*100:.1f}%)")
    
    # Create list of edge types to assign
    edge_types = (
        ["normal"] * normal_count + 
        ["suspicious"] * suspicious_count + 
        ["malicious"] * malicious_count
    )
    
    # Shuffle to randomize assignment
    random.shuffle(edge_types)
    
    # Assign new edge types and colors
    for i, edge in enumerate(edges):
        edge_type = edge_types[i]
        edge["edge_type"] = edge_type
        
        # Assign colors based on type
        if edge_type == "normal":
            edge["color"] = "blue"
        elif edge_type == "suspicious":
            edge["color"] = "orange"
        else:  # malicious
            edge["color"] = "red"
    
    # Update metadata
    if "edge_distribution" not in topology.get("metadata", {}):
        topology.setdefault("metadata", {})
    
    topology["metadata"]["edge_distribution"] = {
        "normal": normal_count,
        "suspicious": suspicious_count, 
        "malicious": malicious_count,
        "total": total_edges
    }
    
    topology["metadata"]["edge_color_scheme"] = {
        "blue": "normal/benign traffic",
        "orange": "suspicious activity",
        "red": "malicious/attack traffic"
    }
    
    # Count actual distribution
    actual_counts = {}
    for edge in edges:
        edge_type = edge["edge_type"]
        actual_counts[edge_type] = actual_counts.get(edge_type, 0) + 1
    
    print(f"✅ Actual distribution applied:")
    for edge_type, count in actual_counts.items():
        percentage = count / total_edges * 100
        print(f"   • {edge_type.capitalize()}: {count} ({percentage:.1f}%)")
    
    # Save updated topology
    print(f"💾 Saving updated {file_path}")
    with open(file_path, 'w') as f:
        json.dump(topology, f, indent=2)
    
    print(f"🎯 Updated edge distribution to realistic network traffic patterns!")

if __name__ == "__main__":
    # Set random seed for reproducible results
    random.seed(42)
    fix_edge_distribution()
