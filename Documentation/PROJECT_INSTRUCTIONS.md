# Network Intrusion Detection 3D Visualization Project

## Project Overview
This project combines machine learning intrusion detection using Universal Hyperbolic Geometry with 3D network visualization in Unreal Engine 5. The goal is to create an immersive 3D environment where network traffic can be visualized and intrusion detection results can be displayed in real-time.

## Project Components

### 1. Data Pipeline
- **Source**: Google Colab dataset for network intrusion detection
- **Model**: Universal Hyperbolic Geometry-based intrusion detection model
- **Initial Scope**: 5-10% of the full dataset for prototyping
- **Data Types**: Network traffic data containing both benign and malicious traffic

### 2. 3D Visualization (Unreal Engine 5)
- **Platform**: Unreal Engine 5 (locally installed on Mac)
- **Visualization**: 3D network topology with nodes (devices) and edges (connections)
- **Interaction**: First-person or free-camera movement through the 3D network space
- **Future Enhancement**: Real-time highlighting of nodes based on model predictions

### 3. Integration Goals
- **Phase 1**: Static 3D network visualization with sample data
- **Phase 2**: Integration with ML model for real-time intrusion detection visualization
- **Phase 3**: Dynamic highlighting of nodes based on traffic classification (benign, DDoS, etc.)

## Technical Requirements

### Data Processing
- [ ] Export relevant network topology data from Google Colab
- [ ] Convert network data to format suitable for Unreal Engine import
- [ ] Create data structure for nodes (devices) and edges (connections)
- [ ] Implement data sampling (5-10% of dataset)
- [ ] Generate JSON exports for GitHub integration
- [ ] Create network topology files for UE5 import

### Unreal Engine 5 Implementation
- [ ] Set up new UE5 project
- [ ] Design 3D network node representation
- [ ] Implement edge/connection visualization
- [ ] Create navigation system for 3D space exploration
- [ ] Develop node interaction system
- [ ] Implement dynamic material/color system for future ML integration

### Model Integration (Future Phase)
- [ ] Create interface between Python ML model and Unreal Engine
- [ ] Implement real-time data streaming
- [ ] Design visual indicators for different traffic types:
  - Benign traffic (e.g., green nodes)
  - DDoS attacks (e.g., red nodes)
  - Other malicious traffic (e.g., orange/yellow nodes)

## Development Phases

### Phase 1: Foundation (Current)
1. Analyze provided Google Colab code and dataset
2. Extract network topology information
3. Create basic 3D network visualization in UE5
4. Implement basic navigation and interaction

### Phase 2: Static Visualization
1. Import real network data into UE5
2. Create comprehensive 3D network layout
3. Implement sophisticated node and edge rendering
4. Add UI for network exploration

### Phase 3: Dynamic Integration
1. Develop communication bridge between ML model and UE5
2. Implement real-time data processing
3. Create dynamic visual feedback system
4. Test with live data streams

## Data Pipeline & GitHub Integration

### GitHub Workflow
- **Data Export**: Export processed data and model results from Google Colab
- **Repository Structure**: Push results to GitHub repository with proper UE5 structure
- **Data Import**: Pull data from GitHub into Unreal Engine 5 locally

### GitHub Repository Structure
```
/ProjectName
    /Source                 # C++ source code
    /Config                 # Configuration files  
    /Plugins                # Project-specific plugins
    /Content                # Unreal Engine assets (.uasset, .umap)
    /RawContent            # Raw assets before UE5 import
    /Data                  # Network data exports from Colab
        /NetworkTopology   # JSON files with node/edge data
        /ModelResults      # ML model predictions and classifications
        /Datasets          # Sample datasets (5-10% subset)
    .gitignore             # UE5-specific gitignore
    .gitattributes         # Git LFS configuration for large files
```

### Supported Data Formats for UE5
- **Data Tables**: CSV/JSON files for structured data import
- **3D Models**: FBX format for node/edge 3D representations
- **Scene Data**: Universal Scene Description (USD) for complex scenes
- **Network Data**: JSON format for real-time API integration

### Git LFS Setup
- Required for handling large UE5 binary files
- Configure for .uasset, .umap, and other large asset files
- Essential for collaborative development

## Technical Stack
- **ML/Data**: Python, Google Colab, Universal Hyperbolic Geometry libraries
- **Version Control**: GitHub with Git LFS for asset management
- **Data Pipeline**: GitHub API for real-time data pulling
- **Visualization**: Unreal Engine 5, Blueprints/C++
- **Data Format**: JSON/CSV for data exchange, FBX for 3D models
- **Platform**: macOS development environment

## Success Criteria
- [ ] Functional 3D network visualization in UE5
- [ ] Smooth navigation through network topology
- [ ] Clear representation of network nodes and connections
- [ ] Foundation for future ML model integration
- [ ] Scalable architecture for real-time data processing

## Notes
- Start with subset of data (5-10%) for initial prototyping
- Focus on creating robust foundation before adding ML integration
- Prioritize performance and user experience in 3D environment
- Plan for future scalability to handle larger datasets
