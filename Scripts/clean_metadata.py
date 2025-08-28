#!/usr/bin/env python3
"""
Script to clean up metadata by removing sample_hostnames and hostname_info sections
"""

import json
from pathlib import Path

def clean_metadata():
    """Remove unnecessary metadata sections"""
    
    # Path to the file
    file_path = Path("Data/network_topology2.json")
    
    print(f"🔄 Loading {file_path}")
    
    # Load topology
    with open(file_path, 'r') as f:
        topology = json.load(f)
    
    # Check what's in metadata currently
    metadata = topology.get('metadata', {})
    print(f"📋 Current metadata keys: {list(metadata.keys())}")
    
    # Remove the sections we don't need
    sections_to_remove = ['sample_hostnames', 'hostname_info']
    removed_sections = []
    
    for section in sections_to_remove:
        if section in metadata:
            del metadata[section]
            removed_sections.append(section)
            print(f"   ✓ Removed: {section}")
    
    if not removed_sections:
        print("   ℹ️  No sections needed to be removed")
    
    print(f"📋 Remaining metadata keys: {list(metadata.keys())}")
    
    # Save updated topology
    print(f"💾 Saving cleaned {file_path}")
    with open(file_path, 'w') as f:
        json.dump(topology, f, indent=2)
    
    print(f"✅ Metadata cleaned! Removed: {', '.join(removed_sections) if removed_sections else 'nothing'}")

if __name__ == "__main__":
    clean_metadata()
