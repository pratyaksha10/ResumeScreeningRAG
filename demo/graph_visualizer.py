import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import os
import re

def generate_applicant_graph(document_list):
    """
    Creates a graph where nodes are Applicants and Skills.
    Edges connect Applicants to the Skills found in their resumes.
    """
    G = nx.Graph()
    
    # Common tech skills to look for (simplified extraction)
    SKILLS_LIST = [
        "Python", "Java", "C++", "JavaScript", "SQL", "React", "Node.js", 
        "Machine Learning", "Data Science", "AWS", "Docker", "Kubernetes",
        "Spark", "Hadoop", "TensorFlow", "PyTorch", "Tableau", "PowerBI",
        "Project Management", "Agile", "Scrum", "UI/UX", "Figma", "DevOps"
    ]
    
    # Color palette
    COLOR_APPLICANT = "#EF553B" # Red (matches 'Retrieved' in vector plot)
    COLOR_SKILL = "#636EFA"     # Blue
    
    for doc in document_list:
        # Extract Applicant ID from the header added in retriever
        # Header format: "Applicant ID 12345\n..."
        header_match = re.search(r"Applicant ID (\w+)", doc)
        if header_match:
            app_id = header_match.group(1)
            app_node_id = f"App_{app_id}"
            G.add_node(app_node_id, label=f"Applicant {app_id}", color=COLOR_APPLICANT, size=25, title=f"Resume of {app_id}")
            
            # Simple skill extraction (case-insensitive)
            content = doc.lower()
            for skill in SKILLS_LIST:
                if skill.lower() in content:
                    G.add_node(skill, label=skill, color=COLOR_SKILL, size=15, title="Skill")
                    G.add_edge(app_node_id, skill)

    # Initialize PyVis Network
    net = Network(
        height="600px", 
        width="100%", 
        bgcolor="#0e1117", 
        font_color="white",
        notebook=False
    )
    
    # Load NetworkX graph
    net.from_nx(G)
    
    # Custom physics for better layout
    net.toggle_physics(True)
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": { "iterations": 150 }
      }
    }
    """)
    
    # Save to HTML file
    path = "graph_visualization.html"
    net.save_graph(path)
    return path

def render_graph_in_streamlit(html_path):
    with open(html_path, 'r', encoding='utf-8') as f:
        html_data = f.read()
    
    # Wrap in a responsive container
    responsive_html = f"""
        <div style="width: 100%; height: 600px; border-radius: 10px; overflow: hidden; border: 1px solid #262730;">
            {html_data}
        </div>
    """
    components.html(html_data, height=620, scrolling=True)
