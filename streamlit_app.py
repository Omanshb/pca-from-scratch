"""
Streamlit App for PCA from Scratch Demonstration

This app demonstrates Principal Component Analysis implemented from scratch
with interactive visualizations and dataset loading capabilities.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import warnings
warnings.filterwarnings('ignore')

# Import our PCA implementation
from pca_from_scratch import PCAFromScratch, standardize_data


def load_sample_datasets():
    """Load built-in sample datasets for demonstration."""
    datasets = {}
    
    # 1. Iris Dataset (classic example)
    try:
        from sklearn.datasets import load_iris
        iris = load_iris()
        datasets['Iris'] = {
            'data': pd.DataFrame(iris.data, columns=iris.feature_names),
            'target': iris.target,
            'target_names': iris.target_names,
            'description': 'Classic iris flower dataset with 4 features and 3 species'
        }
    except ImportError:
        pass
    
    # 2. Wine Dataset
    try:
        from sklearn.datasets import load_wine
        wine = load_wine()
        datasets['Wine'] = {
            'data': pd.DataFrame(wine.data, columns=wine.feature_names),
            'target': wine.target,
            'target_names': wine.target_names,
            'description': 'Wine recognition dataset with chemical analysis features'
        }
    except ImportError:
        pass
    
    # 3. Breast Cancer Dataset
    try:
        from sklearn.datasets import load_breast_cancer
        cancer = load_breast_cancer()
        datasets['Breast Cancer'] = {
            'data': pd.DataFrame(cancer.data, columns=cancer.feature_names),
            'target': cancer.target,
            'target_names': cancer.target_names,
            'description': 'Breast cancer diagnostic features dataset'
        }
    except ImportError:
        pass
    
    return datasets


def load_kaggle_dataset(dataset_path: str):
    """Load dataset from Kaggle path."""
    try:
        import kaggle
        
        # Download dataset
        kaggle.api.dataset_download_files(dataset_path, path='./temp_data', unzip=True)
        
        # Try to find CSV files in the downloaded data
        import os
        import glob
        
        csv_files = glob.glob('./temp_data/*.csv')
        if not csv_files:
            st.error("No CSV files found in the downloaded dataset")
            return None
        
        # Load the first CSV file
        df = pd.read_csv(csv_files[0])
        return df
        
    except Exception as e:
        st.error(f"Error loading Kaggle dataset: {str(e)}")
        return None


def create_2d_scatter_plot(data_transformed, target, target_names, title):
    """Create 2D scatter plot with Plotly."""
    if data_transformed.shape[1] < 2:
        st.error("Need at least 2 components for 2D visualization")
        return None
    
    # Safe class label creation
    class_labels = []
    for t in target:
        if target_names is not None:
            try:
                class_labels.append(str(target_names[int(t)]))
            except (IndexError, ValueError, TypeError):
                class_labels.append(str(t))
        else:
            class_labels.append(str(t))
    
    df_plot = pd.DataFrame({
        'PC1': data_transformed[:, 0],
        'PC2': data_transformed[:, 1],
        'Class': class_labels
    })
    
    fig = px.scatter(
        df_plot, x='PC1', y='PC2', color='Class',
        title=title,
        labels={'PC1': 'First Principal Component', 'PC2': 'Second Principal Component'}
    )
    
    fig.update_layout(
        width=700,
        height=500,
        font=dict(size=12)
    )
    
    return fig


def create_3d_scatter_plot(data_transformed, target, target_names, title):
    """Create 3D scatter plot with Plotly."""
    if data_transformed.shape[1] < 3:
        st.error("Need at least 3 components for 3D visualization")
        return None
    
    # Safe class label creation
    class_labels = []
    for t in target:
        if target_names is not None:
            try:
                class_labels.append(str(target_names[int(t)]))
            except (IndexError, ValueError, TypeError):
                class_labels.append(str(t))
        else:
            class_labels.append(str(t))
    
    df_plot = pd.DataFrame({
        'PC1': data_transformed[:, 0],
        'PC2': data_transformed[:, 1],
        'PC3': data_transformed[:, 2],
        'Class': class_labels
    })
    
    fig = px.scatter_3d(
        df_plot, x='PC1', y='PC2', z='PC3', color='Class',
        title=title,
        labels={
            'PC1': 'First Principal Component',
            'PC2': 'Second Principal Component',
            'PC3': 'Third Principal Component'
        }
    )
    
    fig.update_layout(
        width=700,
        height=600,
        font=dict(size=12)
    )
    
    return fig


def create_explained_variance_plot(pca):
    """Create explained variance plot with Plotly."""
    components_range = list(range(1, len(pca.explained_variance_ratio_) + 1))
    cumulative_var = pca.get_cumulative_variance_ratio()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Individual Explained Variance Ratio', 'Cumulative Explained Variance Ratio')
    )
    
    # Individual explained variance
    fig.add_trace(
        go.Bar(
            x=components_range,
            y=pca.explained_variance_ratio_,
            name='Individual',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Cumulative explained variance
    fig.add_trace(
        go.Scatter(
            x=components_range,
            y=cumulative_var,
            mode='lines+markers',
            name='Cumulative',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Add 95% threshold line
    fig.add_hline(
        y=0.95, line_dash="dash", line_color="red",
        annotation_text="95% threshold",
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Principal Component", row=1, col=1)
    fig.update_xaxes(title_text="Principal Component", row=1, col=2)
    fig.update_yaxes(title_text="Explained Variance Ratio", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Explained Variance Ratio", row=1, col=2)
    
    fig.update_layout(
        height=400,
        title_text="PCA Explained Variance Analysis"
    )
    
    return fig


def create_loadings_plot(pca, feature_names, n_components=2):
    """Create PCA loadings plot."""
    if pca.components_ is None:
        return None
    
    loadings = pca.components_[:n_components].T
    
    if len(feature_names) > 20:  # Too many features for a readable plot
        st.warning("Too many features to display loadings plot clearly. Showing first 20 features.")
        loadings = loadings[:20]
        feature_names = feature_names[:20]
    
    df_loadings = pd.DataFrame(
        loadings,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=feature_names
    )
    
    fig = px.scatter(
        df_loadings, x='PC1', y='PC2' if n_components > 1 else 'PC1',
        title='PCA Loadings Plot',
        labels={'PC1': 'First Principal Component Loading', 'PC2': 'Second Principal Component Loading'}
    )
    
    # Add arrows and labels
    for i, feature in enumerate(feature_names):
        fig.add_annotation(
            x=loadings[i, 0],
            y=loadings[i, 1] if n_components > 1 else 0,
            text=feature,
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="blue"
        )
    
    fig.update_layout(width=700, height=500)
    return fig


def main():
    st.set_page_config(
        page_title="PCA from Scratch Demo",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔍 Principal Component Analysis from Scratch")
    
    # Sidebar for data selection
    st.sidebar.header("📊 Data Selection")
    
    data_source = st.sidebar.radio(
        "Choose data source:",
        ["Built-in Datasets", "Upload CSV", "Kaggle Dataset"]
    )
    
    # Initialize variables
    df = None
    target = None
    target_names = None
    dataset_name = ""
    
    if data_source == "Built-in Datasets":
        datasets = load_sample_datasets()
        
        if not datasets:
            st.error("No built-in datasets available. Please install scikit-learn.")
            return
        
        dataset_choice = st.sidebar.selectbox(
            "Select dataset:",
            list(datasets.keys())
        )
        
        if dataset_choice:
            dataset = datasets[dataset_choice]
            df = dataset['data']
            target = dataset['target']
            target_names = dataset['target_names']
            dataset_name = dataset_choice
            
            st.sidebar.info(f"**{dataset_choice}**\n\n{dataset['description']}")
    
    elif data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            dataset_name = uploaded_file.name
            
            # Let user select target column
            if len(df.columns) > 1:
                target_col = st.sidebar.selectbox(
                    "Select target column (optional):",
                    ["None"] + list(df.columns)
                )
                
                if target_col != "None":
                    target = df[target_col].values
                    df = df.drop(columns=[target_col])
                    target_names = np.unique(target) if len(np.unique(target)) < 10 else None
    
    elif data_source == "Kaggle Dataset":
        st.sidebar.markdown("""
        **Enter Kaggle dataset path:**
        Format: `username/dataset-name`
        
        Example: `uciml/iris`
        """)
        
        kaggle_path = st.sidebar.text_input("Dataset path:")
        
        if kaggle_path and st.sidebar.button("Load Dataset"):
            with st.spinner("Loading Kaggle dataset..."):
                df = load_kaggle_dataset(kaggle_path)
                if df is not None:
                    dataset_name = kaggle_path
                    
                    # Let user select target column
                    if len(df.columns) > 1:
                        target_col = st.sidebar.selectbox(
                            "Select target column (optional):",
                            ["None"] + list(df.columns)
                        )
                        
                        if target_col != "None":
                            target = df[target_col].values
                            df = df.drop(columns=[target_col])
                            target_names = np.unique(target) if len(np.unique(target)) < 10 else None
    
    # If we have data, proceed with PCA
    if df is not None:
        st.header(f"📈 Dataset: {dataset_name}")
        
        # Display basic dataset info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Samples", df.shape[0])
        with col2:
            st.metric("Features", df.shape[1])
        with col3:
            st.metric("Classes", len(np.unique(target)) if target is not None else "N/A")
        
        # Show dataset preview
        with st.expander("📋 Dataset Preview"):
            st.dataframe(df.head(10))
        
        # Data preprocessing options
        st.subheader("⚙️ Preprocessing Options")
        
        col1, col2 = st.columns(2)
        with col1:
            standardize = st.checkbox("Standardize data", value=True, 
                                    help="Recommended for features with different scales")
        with col2:
            n_components = st.slider("Number of components", 
                                   min_value=2, 
                                   max_value=min(df.shape[0], df.shape[1], 10),
                                   value=min(3, df.shape[1]))
        
        # Prepare data
        X = df.select_dtypes(include=[np.number]).values
        
        if X.shape[1] < 2:
            st.error("Need at least 2 numerical features for PCA")
            return
        
        if standardize:
            X, _, _ = standardize_data(X)
        
        # Perform PCA
        with st.spinner("Performing PCA..."):
            pca = PCAFromScratch(n_components=n_components)
            X_transformed = pca.fit_transform(X)
        
        # Results section
        st.header("📊 PCA Results")
        
        # Explained variance
        st.subheader("Explained Variance Analysis")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            fig_variance = create_explained_variance_plot(pca)
            st.plotly_chart(fig_variance, use_container_width=True)
            st.info("""
            **Scree Plot Explanation:**
            - **Left plot**: Shows how much variance each PC explains individually
            - **Right plot**: Shows cumulative variance captured as you add more PCs
            - **Look for the 'elbow'**: Point where variance drops sharply
            - **95% rule**: Keep enough PCs to capture 95% of total variance
            """)
        
        with col2:
            st.write("**Explained Variance Ratios:**")
            for i, ratio in enumerate(pca.explained_variance_ratio_):
                st.write(f"PC{i+1}: {ratio:.4f} ({ratio*100:.2f}%)")
            
            cumulative = pca.get_cumulative_variance_ratio()
            st.write(f"\n**Cumulative Variance (first {n_components} components):** {cumulative[-1]:.4f} ({cumulative[-1]*100:.2f}%)")
            
            st.info("""
            **Interpretation Guide:**
            - Each PC captures a portion of the data's variance
            - PC1 always captures the most variance
            - PCs are ordered by importance (variance explained)
            - Use this to decide how many components to keep
            """)
        
        # Visualizations
        st.subheader("🎯 Data Visualizations")
        
        viz_tabs = st.tabs(["2D Scatter", "3D Scatter", "Feature Loadings"])
        
        with viz_tabs[0]:
            if target is not None:
                fig_2d = create_2d_scatter_plot(X_transformed, target, target_names, 
                                              f"PCA Visualization - {dataset_name}")
                st.plotly_chart(fig_2d, use_container_width=True)
                st.info("""
                **2D PCA Scatter Plot:**
                - Shows your high-dimensional data projected onto the first 2 principal components
                - Each point represents one sample from your dataset
                - Colors represent different classes/categories
                - **Clusters**: Points of the same color clustering together indicates good class separation
                - **PC1 (x-axis)**: Captures the most variance in your data
                - **PC2 (y-axis)**: Captures the second most variance
                """)
            else:
                st.info("No target variable available for coloring. Upload data with target column or use built-in datasets.")
        
        with viz_tabs[1]:
            if n_components >= 3:
                if target is not None:
                    fig_3d = create_3d_scatter_plot(X_transformed, target, target_names, 
                                                  f"3D PCA Visualization - {dataset_name}")
                    st.plotly_chart(fig_3d, use_container_width=True)
                    st.info("""
                    **3D PCA Scatter Plot:**
                    - Shows your data in the space of the first 3 principal components
                    - **Interact**: Click and drag to rotate the 3D plot
                    - **PC3 (z-axis)**: Adds the third most important dimension
                    - Better class separation may be visible in 3D vs 2D
                    - More comprehensive view of your data's structure
                    """)
                else:
                    st.info("No target variable available for coloring. Upload data with target column or use built-in datasets.")
            else:
                st.info("Select at least 3 components for 3D visualization")
        
        with viz_tabs[2]:
            if n_components >= 2:
                feature_names = df.select_dtypes(include=[np.number]).columns.tolist()
                fig_loadings = create_loadings_plot(pca, feature_names, min(n_components, 2))
                if fig_loadings:
                    st.plotly_chart(fig_loadings, use_container_width=True)
                    st.info("""
                    **PCA Loadings Plot:**
                    - Shows how much each **original feature** contributes to each principal component
                    - **Arrows**: Point in the direction of feature influence
                    - **Length**: Longer arrows = stronger influence on that PC
                    - **Angle**: Features pointing in similar directions are correlated
                    - **Use for feature selection**: Features with high loadings in important PCs are most important
                    """)
            else:
                st.info("Select at least 2 components for loadings plot")
        
    else:
        st.info("👆 Please select a data source from the sidebar to get started!")
        
        # Show some information about PCA
        st.markdown("""
        ## 🎓 About PCA (Principal Component Analysis)
        
        PCA is a dimensionality reduction technique that:
        
        1. **Finds the directions of maximum variance** in your data
        2. **Projects data onto these directions** (principal components)
        3. **Reduces dimensionality** while preserving most information
        4. **Helps visualize** high-dimensional data
        
        ### 🔧 How it works:
        1. **Standardize** the data (optional but recommended)
        2. **Compute covariance matrix** of the features
        3. **Find eigenvalues and eigenvectors** of the covariance matrix
        4. **Sort by eigenvalues** (largest first)
        5. **Transform data** using selected eigenvectors
        
        ### 🎯 Use cases:
        - **Data visualization** (reduce to 2D/3D)
        - **Noise reduction** 
        - **Feature extraction**
        - **Data compression**
        - **Exploratory data analysis**
        """)


if __name__ == "__main__":
    main() 