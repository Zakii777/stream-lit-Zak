"""
Exploratory Data Analysis (EDA) App using Streamlit
This app loads the Iris dataset and provides interactive visualizations
for data exploration and analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Iris EDA App", layout="wide")

# App title and description
st.title("🌸 Iris Dataset Exploratory Data Analysis")
st.markdown(
    """
    This interactive app allows you to explore the Iris dataset with various
    visualizations and statistical summaries.
    """
)

# Load the Iris dataset
@st.cache_data
def load_data():
    """Load the Iris dataset from scikit-learn."""
    iris_data = load_iris()
    
    # Create a DataFrame from the dataset
    df_iris = pd.DataFrame(
        data=iris_data.data,
        columns=iris_data.feature_names
    )
    
    # Add the target column (species names)
    df_iris['species'] = iris_data.target_names[iris_data.target]
    
    return df_iris, iris_data.feature_names


# Load data
df = load_data()[0]
numeric_columns = list(load_data()[1])

# Create tabs for better organization
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Overview",
    "📈 Statistics",
    "📉 Histogram",
    "🔵 Scatter Plot"
])

# ===== TAB 1: Data Overview =====
with tab1:
    st.subheader("First Rows of the Dataset")
    
    # Display number of rows slider
    num_rows = st.slider(
        "Number of rows to display:",
        min_value=1,
        max_value=len(df),
        value=5
    )
    
    # Display the data
    st.dataframe(df.head(num_rows))
    
    # Dataset info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", len(df))
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric("Unique Species", df['species'].nunique())
    
    # Display species distribution
    st.subheader("Species Distribution")
    species_counts = df['species'].value_counts()
    st.bar_chart(species_counts)


# ===== TAB 2: Summary Statistics =====
with tab2:
    st.subheader("Descriptive Statistics")
    
    # Overall statistics
    st.write("**Overall Dataset Statistics:**")
    st.dataframe(df.describe())
    
    # Statistics by species
    st.write("**Statistics by Species:**")
    
    # Create columns for better layout
    selected_species = st.multiselect(
        "Select species to compare:",
        options=df['species'].unique(),
        default=df['species'].unique()
    )
    
    if selected_species:
        df_filtered = df[df['species'].isin(selected_species)]
        st.dataframe(df_filtered.groupby('species')[numeric_columns].describe())
    
    # Correlation matrix
    st.write("**Correlation Matrix (Numeric Columns):**")
    correlation_matrix = df[numeric_columns].corr()
    st.dataframe(correlation_matrix)


# ===== TAB 3: Histogram =====
with tab3:
    st.subheader("Histogram Analysis")
    
    # Column selection
    selected_column = st.selectbox(
        "Select a column for histogram:",
        options=numeric_columns
    )
    
    # Number of bins slider
    num_bins = st.slider(
        "Number of bins:",
        min_value=5,
        max_value=50,
        value=15
    )
    
    # Create histogram with species differentiation
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for species in df['species'].unique():
        species_data = df[df['species'] == species][selected_column]
        ax.hist(
            species_data,
            bins=num_bins,
            alpha=0.6,
            label=species
        )
    
    ax.set_xlabel(selected_column)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of {selected_column} by Species")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    st.pyplot(fig)


# ===== TAB 4: Scatter Plot =====
with tab4:
    st.subheader("Scatter Plot Analysis")
    
    # Create two columns for x and y axis selection
    col1, col2 = st.columns(2)
    
    with col1:
        x_column = st.selectbox(
            "Select X-axis column:",
            options=numeric_columns,
            key="x_axis"
        )
    
    with col2:
        y_column = st.selectbox(
            "Select Y-axis column:",
            options=numeric_columns,
            key="y_axis",
            index=1
        )
    
    # Create scatter plot with species as different colors
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'setosa': '#ff7f0e', 'versicolor': '#2ca02c', 'virginica': '#1f77b4'}
    
    for species in df['species'].unique():
        species_data = df[df['species'] == species]
        ax.scatter(
            species_data[x_column],
            species_data[y_column],
            label=species,
            alpha=0.7,
            s=100,
            color=colors.get(species, '#808080')
        )
    
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_title(f"{x_column} vs {y_column}")
    ax.legend()
    ax.grid(alpha=0.3)
    
    st.pyplot(fig)


# Footer
st.markdown("---")
st.markdown(
    """
    *Created with Streamlit | Data Source: Scikit-learn Iris Dataset*
    """
)
