import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
import streamlit as st

def get_all_vectors(vectordb):
    """
    Extracts all vectors from the FAISS index.
    """
    ntotal = vectordb.index.ntotal
    # Reconstruct all vectors
    vectors = vectordb.index.reconstruct_n(0, ntotal)
    return vectors

def visualize_vectors(vectordb, query_vector=None, retrieved_ids=None):
    """
    Reduces dimensions of vectors and plots them using Plotly.
    """
    vectors = get_all_vectors(vectordb)
    
    # Add query vector if provided
    if query_vector is not None:
        all_vectors = np.vstack([vectors, query_vector.reshape(1, -1)])
    else:
        all_vectors = vectors
        
    # Perform PCA
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(all_vectors)
    
    # Prepare DataFrame for plotting
    ntotal = vectordb.index.ntotal
    df_plot = pd.DataFrame(reduced_vectors[:ntotal], columns=['x', 'y'])
    
    # Map index to applicant IDs
    labels = []
    types = []
    
    # Get applicant IDs from docstore
    docstore = vectordb.docstore
    index_to_id = vectordb.index_to_docstore_id
    
    retrieved_ids_str = [str(rid) for rid in retrieved_ids] if retrieved_ids else []
    
    for i in range(ntotal):
        doc_id = index_to_id[i]
        doc = docstore.search(doc_id)
        applicant_id = str(doc.metadata.get('ID', 'Unknown'))
        labels.append(f"Applicant {applicant_id}")
        
        if applicant_id in retrieved_ids_str:
            types.append('Retrieved')
        else:
            types.append('Resume')
            
    df_plot['label'] = labels
    df_plot['type'] = types
    
    # Add query point
    if query_vector is not None:
        query_row = pd.DataFrame({
            'x': [reduced_vectors[-1, 0]],
            'y': [reduced_vectors[-1, 1]],
            'label': ['Current Query'],
            'type': ['Query']
        })
        df_plot = pd.concat([df_plot, query_row], ignore_index=True)
        
    # Create Plotly figure
    fig = px.scatter(
        df_plot, 
        x='x', 
        y='y', 
        color='type',
        hover_name='label',
        title="Vector Space Visualization (PCA 2D Projection)",
        labels={'x': 'Semantic Dimension 1 (PCA)', 'y': 'Semantic Dimension 2 (PCA)', 'type': 'Point Type'},
        color_discrete_map={
            'Resume': '#636EFA',
            'Retrieved': '#EF553B',
            'Query': '#00CC96'
        },
        template="plotly_dark"
    )
    
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(
        legend_title_text='Point Type',
        xaxis_title="Semantic Dimension 1 (PCA)",
        yaxis_title="Semantic Dimension 2 (PCA)"
    )
    
    return fig
