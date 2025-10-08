"""
Visualization utilities for POCUS-AI.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, List, Any, Optional, Union


def visualize_features(features: Dict[str, Any], 
                       selected_features: Optional[List[str]] = None,
                       figsize: tuple = (12, 8)):
    """
    Visualize radiomics features
    
    Args:
        features: Dictionary of radiomics features
        selected_features: List of features to visualize (if None, use common features)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Filter out diagnostic features
    filtered_features = {k: v for k, v in features.items() 
                         if not k.startswith('diagnostics_') and isinstance(v, (int, float))}
    
    # If no features specified, use common ones or first 10
    if selected_features is None:
        common_features = [
            'firstorder_Mean', 'firstorder_Median', 'firstorder_Entropy',
            'firstorder_Energy', 'firstorder_Kurtosis', 'firstorder_Skewness',
            'glcm_Contrast', 'glcm_Correlation', 'glcm_JointEntropy',
            'shape2D_Area', 'shape2D_Perimeter', 'shape2D_Sphericity'
        ]
        selected_features = [f for f in common_features if f in filtered_features]
        
        # If still no features, use first 10
        if not selected_features:
            selected_features = list(filtered_features.keys())[:10]
    
    # Filter to only selected features
    features_to_plot = {k: v for k, v in filtered_features.items() if k in selected_features}
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create barplot
    feature_names = list(features_to_plot.keys())
    feature_values = list(features_to_plot.values())
    
    # Group features by type
    feature_types = {}
    for name in feature_names:
        if '_' in name:
            feat_type = name.split('_')[0]
            if feat_type not in feature_types:
                feature_types[feat_type] = []
            feature_types[feat_type].append(name)
    
    # Create color map
    colors = plt.cm.tab10(np.linspace(0, 1, len(feature_types)))
    color_map = {}
    for i, feat_type in enumerate(feature_types.keys()):
        color_map[feat_type] = colors[i]
    
    # Create bar colors
    bar_colors = [color_map[name.split('_')[0]] for name in feature_names]
    
    # Plot
    bars = ax.bar(feature_names, feature_values, color=bar_colors)
    
    # Add labels and title
    ax.set_ylabel('Feature Value')
    ax.set_title('Radiomics Features')
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha='right')
    
    # Add legend
    legend_elements = [plt.Rectangle((0,0), 1, 1, color=color_map[feat_type], 
                                     label=feat_type) 
                       for feat_type in feature_types.keys()]
    ax.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    
    return fig


def compare_features(feature_sets: Dict[str, Dict[str, Any]],
                     selected_features: Optional[List[str]] = None,
                     figsize: tuple = (14, 10)):
    """
    Compare radiomics features across multiple sets
    
    Args:
        feature_sets: Dictionary mapping set names to feature dictionaries
        selected_features: List of features to compare
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Prepare data for visualization
    data = []
    
    for set_name, features in feature_sets.items():
        # Filter out diagnostic features
        filtered_features = {k: v for k, v in features.items() 
                             if not k.startswith('diagnostics_') and isinstance(v, (int, float))}
        
        # If no features specified, find common ones across all sets
        if selected_features is None:
            if not data:  # First set
                potential_features = list(filtered_features.keys())
            else:
                potential_features = [f for f in potential_features if f in filtered_features]
                
            # Take first 10 common features if there are many
            if len(potential_features) > 10:
                selected_features = potential_features[:10]
            else:
                selected_features = potential_features
        
        # Add to data
        for feature, value in filtered_features.items():
            if feature in selected_features:
                data.append({
                    'Set': set_name,
                    'Feature': feature,
                    'Value': value
                })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    if len(df) == 0:
        print("No common features found across sets")
        return None
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # Create grouped bar plot
    sns.barplot(x='Feature', y='Value', hue='Set', data=df, ax=axes[0])
    axes[0].set_title('Comparison of Radiomics Features')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
    
    # Create heatmap
    pivot_df = df.pivot(index='Set', columns='Feature', values='Value')
    
    # Normalize for better visualization
    normalized_df = pivot_df.copy()
    for col in normalized_df.columns:
        min_val = normalized_df[col].min()
        max_val = normalized_df[col].max()
        if max_val > min_val:
            normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
    
    sns.heatmap(normalized_df, annot=True, cmap='viridis', fmt='.2f', ax=axes[1])
    axes[1].set_title('Normalized Feature Comparison')
    
    plt.tight_layout()
    
    return fig