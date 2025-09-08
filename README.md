# Unreal-UHG: 3D Network Intrusion Detection Visualization

This project combines Universal Hyperbolic Geometry (UHG) machine learning with Unreal Engine 5 to create an immersive 3D visualization of network intrusion detection data.

## Project Overview

- **Machine Learning**: UHG-based GraphSAGE model for intrusion detection
- **Dataset**: CIC network traffic data with attack classifications
- **Visualization**: Real-time 3D network topology in Unreal Engine 5
- **Data Pipeline**: Google Colab → GitHub → Unreal Engine 5

## Repository Structure

```
Unreal-UHG/
├── Data/                    # Network data exports from ML model
├── Documentation/           # Project documentation and instructions
├── Scripts/                 # Google Colab notebooks and export scripts
├── UnrealProject/          # Unreal Engine 5 project files
└── UHG IDS Model/          # Original ML model code and results
```

## Attack Type Visualization

- 🟢 **Green Nodes**: Benign network traffic
- 🔴 **Red Nodes**: DDoS/DoS attacks
- 🟠 **Orange Nodes**: Other malicious traffic
- 🔵 **Blue Edges**: Normal connections
- 🔴 **Red Edges**: Suspicious connections

## Getting Started

1. **Data Export**: Run the Google Colab notebook in `Scripts/`
2. **UE5 Project**: Open the Unreal Engine 5 project in `UnrealProject/`
3. **Data Import**: The UE5 project automatically pulls data from this GitHub repository

## Technologies Used

- **ML Framework**: PyTorch Geometric with Universal Hyperbolic Geometry
- **Game Engine**: Unreal Engine 5
- **Data Pipeline**: GitHub API integration
- **Languages**: Python (ML), Blueprint/C++ (UE5)

## Future Enhancements

- Real-time data streaming
- Interactive node exploration
- Network traffic flow animation
- VR support for immersive exploration

---

*This project demonstrates the intersection of advanced machine learning and interactive 3D visualization for cybersecurity applications.*
