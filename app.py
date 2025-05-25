"""
US Road Network Analysis with Neo4j and Interactive Dashboard
MSDA9215: Big Data Analytics - Neo4j Hands-on Assignment

This script performs comprehensive analysis of US road network data using Neo4j
and creates an interactive dashboard with Plotly Dash.
"""

import pandas as pd
import numpy as np
from neo4j import GraphDatabase
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import networkx as nx
import math
from collections import Counter
import time
import logging
from typing import List, Dict, Tuple, Optional
from dash import State 

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class USRoadNetworkAnalyzer:
    """Main class for analyzing US road network data with Neo4j"""
    
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="auca_assignment"):
        """Initialize Neo4j connection"""
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Successfully connected to Neo4j database")
            self.data_loaded = False
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise ConnectionError(f"Cannot connect to Neo4j at {uri}. Please ensure Neo4j is running and credentials are correct.")
        
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
    
    def clear_database(self):
        """Clear all nodes and relationships from the database"""
        with self.driver.session() as session:
            # Check if there's existing data
            result = session.run("MATCH (n) RETURN count(n) as node_count")
            node_count = result.single()["node_count"]
            
            if node_count > 0:
                logger.info(f"Clearing {node_count} existing nodes from database...")
                session.run("MATCH (n) DETACH DELETE n")
                logger.info("Database cleared")
            else:
                logger.info("Database is already empty")
    
    def load_data_from_file(self, filepath: str, chunk_size: int = 1000):
        """
        Load road network data from file into Neo4j with chunking for performance
        
        Format:
        Line 1: num_vertices num_edges
        Lines 2 to num_vertices+1: vertex_id x_coord y_coord
        Remaining lines: vertex1 vertex2 (edges)
        """
        try:
            with open(filepath, 'r') as file:
                lines = file.readlines()
            
            # Parse header
            header = lines[0].strip().split()
            num_vertices = int(header[0])
            num_edges = int(header[1])
            
            logger.info(f"Loading {num_vertices} vertices and {num_edges} edges")
            
            # Clear existing data
            self.clear_database()
            
            # Create constraints and indexes for performance (handle existing ones gracefully)
            with self.driver.session() as session:
                try:
                    session.run("CREATE CONSTRAINT intersection_id IF NOT EXISTS FOR (i:Intersection) REQUIRE i.id IS UNIQUE")
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        logger.warning(f"Constraint creation issue: {e}")
                
                try:
                    session.run("CREATE INDEX intersection_coords IF NOT EXISTS FOR (i:Intersection) ON (i.x, i.y)")
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        logger.warning(f"Index creation issue: {e}")
                
                logger.info("Database constraints and indexes verified")
            
            # Load vertices in chunks
            vertices_data = []
            for i in range(1, num_vertices + 1):
                parts = lines[i].strip().split()
                vertex_id = int(parts[0])
                x_coord = float(parts[1])
                y_coord = float(parts[2])
                vertices_data.append({'id': vertex_id, 'x': x_coord, 'y': y_coord})
            
            # Insert vertices in chunks
            self._insert_vertices_chunked(vertices_data, chunk_size)
            
            # Load edges in chunks
            edges_data = []
            for i in range(num_vertices + 1, len(lines)):
                if lines[i].strip():
                    parts = lines[i].strip().split()
                    if len(parts) >= 2:
                        vertex1 = int(parts[0])
                        vertex2 = int(parts[1])
                        edges_data.append({'v1': vertex1, 'v2': vertex2})
            
            # Insert edges in chunks
            self._insert_edges_chunked(edges_data, chunk_size)
            
            self.data_loaded = True
            logger.info("Data loading completed successfully")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _insert_vertices_chunked(self, vertices_data: List[Dict], chunk_size: int):
        """Insert vertices in chunks for better performance"""
        with self.driver.session() as session:
            for i in range(0, len(vertices_data), chunk_size):
                chunk = vertices_data[i:i + chunk_size]
                session.run("""
                    UNWIND $vertices AS vertex
                    CREATE (i:Intersection {
                        id: vertex.id,
                        x: vertex.x,
                        y: vertex.y
                    })
                """, vertices=chunk)
                logger.info(f"Inserted vertices chunk {i//chunk_size + 1}/{math.ceil(len(vertices_data)/chunk_size)}")
    
    def _insert_edges_chunked(self, edges_data: List[Dict], chunk_size: int):
        """Insert edges in chunks for better performance"""
        with self.driver.session() as session:
            for i in range(0, len(edges_data), chunk_size):
                chunk = edges_data[i:i + chunk_size]
                session.run("""
                    UNWIND $edges AS edge
                    MATCH (i1:Intersection {id: edge.v1})
                    MATCH (i2:Intersection {id: edge.v2})
                    CREATE (i1)-[r:ROAD {
                        distance: sqrt((i1.x - i2.x)^2 + (i1.y - i2.y)^2)
                    }]->(i2)
                """, edges=chunk)
                logger.info(f"Inserted edges chunk {i//chunk_size + 1}/{math.ceil(len(edges_data)/chunk_size)}")
    
    def get_basic_stats(self) -> Dict:
        """Get basic network statistics"""
        with self.driver.session() as session:
            # Total intersections
            result = session.run("MATCH (i:Intersection) RETURN count(i) as total_intersections")
            total_intersections = result.single()["total_intersections"]
            
            # Total roads
            result = session.run("MATCH ()-[r:ROAD]-() RETURN count(r) as total_roads")
            total_roads = result.single()["total_roads"]
            result = session.run("MATCH (i:Intersection)-[:ROAD]->(j:Intersection) WHERE i.id < j.id RETURN count(*) as total_roads")
            result = session.run("MATCH (i:Intersection)-[:ROAD]->(j:Intersection) RETURN count(*) as total_roads")

            total_roads = result.single()["total_roads"]
            
            # Average degree
            result = session.run("""
                MATCH (i:Intersection)
                RETURN avg(size([(i)-[:ROAD]-() | 1])) as avg_degree
            """)
            avg_degree = result.single()["avg_degree"]
            
            return {
                'total_intersections': total_intersections,
                'total_roads': total_roads,
                'average_degree': round(avg_degree, 2) if avg_degree else 0
            }
    
    def find_shortest_path(self, source_id: int, target_id: int) -> Dict:
        with self.driver.session() as session:
            result = session.run("""
                MATCH (start:Intersection {id: $source_id}), (end:Intersection {id: $target_id})
                MATCH path = shortestPath((start)-[:ROAD*..100]-(end))
                RETURN [n IN nodes(path) | n.id] as path_ids, length(path) as hops
            """, source_id=source_id, target_id=target_id)
            record = result.single()
            if record:
                return {
                    'path_ids': record['path_ids'],
                    'path_length': record['hops']
                }
            return {'path_ids': [], 'path_length': 0}
    
    def get_high_degree_intersections(self, threshold: int = 3) -> List[Dict]:
        with self.driver.session() as session:
            result = session.run("""
                MATCH (i:Intersection)
                OPTIONAL MATCH (i)-[:ROAD]-()
                WITH i, count(*) AS degree
                WHERE degree > $threshold
                RETURN i.id AS id, i.x AS x, i.y AS y, degree
                ORDER BY degree asc
            """, threshold=threshold)
            return [dict(record) for record in result]
        
    def get_all_node_degrees(self) -> List[Dict]:
        with self.driver.session() as session:
            result = session.run("""
                MATCH (i:Intersection)-[:ROAD]-(j:Intersection)
                WITH i.id AS id, count(DISTINCT j) AS degree
                RETURN id, degree
                ORDER BY degree DESC
            """)
            return [dict(r) for r in result]

    
    def calculate_betweenness_centrality(self, sample_size: int = 100) -> List[Dict]:
        """
        Calculate betweenness centrality for a sample of nodes
        (Full calculation would be too expensive for large networks)
        """
        with self.driver.session() as session:
            # Get sample of nodes
            result = session.run("""
                MATCH (i:Intersection)
                WITH i, rand() as r
                ORDER BY r
                LIMIT $sample_size
                RETURN collect(i.id) as node_ids
            """, sample_size=sample_size)
            
            node_ids = result.single()["node_ids"]
            
            # Calculate betweenness centrality using NetworkX for efficiency
            # First, get the subgraph
            subgraph_result = session.run("""
                MATCH (i1:Intersection)-[r:ROAD]-(i2:Intersection)
                WHERE i1.id IN $node_ids AND i2.id IN $node_ids
                RETURN i1.id as source, i2.id as target, r.distance as weight
            """, node_ids=node_ids)
            
            # Build NetworkX graph
            G = nx.Graph()
            for record in subgraph_result:
                G.add_edge(record['source'], record['target'], weight=record['weight'])
            
            # Calculate betweenness centrality
            if len(G.nodes()) > 0:
                centrality = nx.betweenness_centrality(G, weight='weight')
                return [{'id': node_id, 'centrality': cent} for node_id, cent in centrality.items()]
            
            return []
    
    def get_degree_distribution(self) -> Dict:
        """Get degree distribution of intersections"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (i:Intersection)
                WITH i, size([(i)-[:ROAD]-() | 1]) as degree
                RETURN degree, count(*) as count
                ORDER BY degree
            """)
            
            distribution = {}
            for record in result:
                distribution[record['degree']] = record['count']
            
            return distribution
    
    def get_top_connected_intersections(self, limit: int = 10) -> List[Dict]:
        """Get top N most connected intersections"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (i:Intersection)
                WITH i, size([(i)-[:ROAD]-() | 1]) as degree
                WHERE degree > 0
                RETURN i.id as id, i.x as x, i.y as y, degree
                ORDER BY degree DESC
                LIMIT $limit
            """, limit=limit)
            
            return [dict(record) for record in result]
    
    def categorize_intersections_by_degree(self) -> Dict:
        """Categorize intersections by connectivity level"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (i:Intersection)
                WITH i, size([(i)-[:ROAD]-() | 1]) as degree
                WHERE degree IS NOT NULL
                RETURN 
                    count(CASE WHEN degree IS NOT NULL AND degree <= 2 THEN 1 END) as low_connectivity,
                    count(CASE WHEN degree IS NOT NULL AND degree > 2 AND degree <= 5 THEN 1 END) as medium_connectivity,
                    count(CASE WHEN degree IS NOT NULL AND degree > 5 THEN 1 END) as high_connectivity
            """)
            
            record = result.single()
            return {
                'Low (≤2)': record['low_connectivity'],
                'Medium (3-5)': record['medium_connectivity'],
                'High (>5)': record['high_connectivity']
            }
    
    def get_geographic_distribution(self) -> List[Dict]:
        """Get geographic distribution of intersections"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (i:Intersection)
                RETURN i.x as x, i.y as y, i.id as id,
                       size([(i)-[:ROAD]-() | 1]) as degree
                LIMIT 5000
            """)
            
            return [dict(record) for record in result]
    def get_sample_intersection_ids(self, count: int = 20) -> List[int]:
        """Get a sample of intersection IDs for dropdown options"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (i:Intersection)
                WITH i, size([(i)-[:ROAD]-() | 1]) as degree
                WHERE degree > 0
                RETURN i.id as id
                ORDER BY degree DESC
                LIMIT $count
            """, count=count)
            
            return [record['id'] for record in result]

# Dashboard Creation
def create_dashboard(analyzer: USRoadNetworkAnalyzer):
    """Create interactive dashboard using Dash"""
    
    # Initialize Dash app
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    # Custom CSS
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>US Road Network Analysis Dashboard</title>
            {%favicon%}
            {%css%}
            <style>
                .dashboard-header {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 2rem;
                    margin-bottom: 2rem;
                    border-radius: 0px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }
                .dashboard-header {
                    background: #0068aa;
                    color: white;
                    padding: 2rem;
                    margin-bottom: 2rem;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                }
                .dashboard-header img.logo {
                    height: 50px;
                    margin-right: 1rem;
                    width: 50px;
                }
                .metric-card {
                    background: white;
                    border-radius: 10px;
                    padding: 1.5rem;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                    border-left: 4px solid #667eea;
                    margin-bottom: 1rem;
                }
                .chart-container {
                    background: white;
                    border-radius: 10px;
                    padding: 1rem;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                    margin-bottom: 2rem;
                }
                
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''
    
    # Get data for dashboard
    try:
        basic_stats = analyzer.get_basic_stats()
        degree_dist = analyzer.get_degree_distribution()
        top_connected = analyzer.get_top_connected_intersections(10)
        categories = analyzer.categorize_intersections_by_degree()
        geographic_data = analyzer.get_geographic_distribution()
        high_degree = analyzer.get_high_degree_intersections(5)
        sample_ids = analyzer.get_sample_intersection_ids(50)
        
        # Create visualizations
        
        # 1. Degree Distribution Bar Chart
        degree_fig = px.bar(
            x=list(degree_dist.keys()),
            y=list(degree_dist.values()),
            title="Degree Distribution of Intersections",
            labels={'x': 'Degree (Number of Roads)', 'y': 'Number of Intersections'},
            color=list(degree_dist.values()),
            color_continuous_scale='viridis'
        )
        degree_fig.update_layout(
            title_font_size=16,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        # 2. Top Connected Intersections
        top_connected_fig = px.bar(
            x=[f"ID {item['id']}" for item in top_connected],
            y=[item['degree'] for item in top_connected],
            title="Top 10 Most Connected Intersections",
            labels={'x': 'Intersection ID', 'y': 'Number of Connections'},
            color=[item['degree'] for item in top_connected],
            color_continuous_scale='plasma'
        )
        top_connected_fig.update_layout(
            title_font_size=16,
            xaxis_tickangle=-45,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        # 3. Connectivity Categories Pie Chart
        categories_fig = px.pie(
            values=list(categories.values()),
            names=list(categories.keys()),
            title="Intersection Categories by Connectivity Level",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        categories_fig.update_layout(
            title_font_size=16,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        # 4. Geographic Distribution Scatter Plot
        geo_fig = px.scatter(
            x=[item['x'] for item in geographic_data],
            y=[item['y'] for item in geographic_data],
            color=[item['degree'] for item in geographic_data],
            size=[max(1, item['degree']) for item in geographic_data],
            title="Geographic Distribution of Intersections",
            labels={'x': 'X Coordinate', 'y': 'Y Coordinate'},
            color_continuous_scale='viridis',
            hover_data={'ID': [item['id'] for item in geographic_data]}
        )
        geo_fig.update_layout(
            title_font_size=16,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        # 5. Network Statistics Gauge Charts
        gauges_fig = make_subplots(
            rows=1, cols=3,
            # subplot_titles=('Network Density', 'Average Degree', 'Connectivity Index'),
            subplot_titles=(" ", " ", " "),
            specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]]
        )
        
        # Network density (simplified calculation)
        max_possible_edges = basic_stats['total_intersections'] * (basic_stats['total_intersections'] - 1) / 2
        network_density = (basic_stats['total_roads'] / max_possible_edges) * 100 if max_possible_edges > 0 else 0
        
        gauges_fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=network_density,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Density %"},
            gauge={'axis': {'range': [None, 1]},
                   'bar': {'color': "darkblue"},
                   'steps': [{'range': [0, 0.5], 'color': "lightgray"},
                            {'range': [0.5, 1], 'color': "gray"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                               'thickness': 0.75, 'value': 0.8}}),
            row=1, col=1)
        
        gauges_fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=basic_stats['average_degree'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Avg Degree"},
            gauge={'axis': {'range': [None, 10]},
                   'bar': {'color': "darkgreen"},
                   'steps': [{'range': [0, 3], 'color': "lightgray"},
                            {'range': [3, 10], 'color': "gray"}]}),
            row=1, col=2)
        
        # Connectivity index (percentage of nodes with degree > 2)
        high_conn_pct = (categories.get('Medium (3-5)', 0) + categories.get('High (>5)', 0)) / basic_stats['total_intersections'] * 100
        
        gauges_fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=high_conn_pct,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "High Conn %"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "darkred"},
                   'steps': [{'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 100], 'color': "gray"}]}),
            row=1, col=3)
        
        gauges_fig.update_layout(
            height=300,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
    except Exception as e:
        logger.error(f"Error creating dashboard data: {str(e)}")
        # Create empty figures as fallback
        degree_fig = go.Figure()
        degree_fig.add_annotation(text="Data not available", x=0.5, y=0.5, showarrow=False)
        
        top_connected_fig = go.Figure()
        categories_fig = go.Figure()
        geo_fig = go.Figure()
        gauges_fig = go.Figure()
        basic_stats = {'total_intersections': 0, 'total_roads': 0, 'average_degree': 0}
    
    # Layout
    app.layout = dbc.Container([
        # Header
        # html.Img(src="https://github.com/CelestinNiyomugabo/neo4j/blob/main/logo.png?raw=true", className="logo"), 
        # html.Div([
        #     html.H1("🛣️ US Road Network Analysis Dashboard", className="text-center mb-0"),
        #     html.P("Comprehensive analysis of intersection connectivity and network topology - Celestin, Claude, and Vincent", 
        #            className="text-center mb-0", style={'fontSize': '1rem', 'opacity': '0.9'})
        # ], className="dashboard-header"),
        html.Div([
            html.Div([
                html.Img(src="https://github.com/CelestinNiyomugabo/neo4j/blob/main/logo.png?raw=true", style={"height": "150px", "marginRight": "2rem"}),
                html.Div([
                    html.H1("US Road Network Analysis Dashboard", className="mb-0"),
                    html.P("Comprehensive analysis of intersection connectivity and network topology",
                        className="mb-0", style={"fontSize": "1.2rem", "opacity": "0.9"})
                ])
            ], style={"display": "flex", "alignItems": "center"})
        ], className="dashboard-header"),
        
        
        # Key Metrics Row
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H3(f"{basic_stats['total_intersections']:,}", className="text-primary mb-1"),
                    html.P("Total Intersections", className="mb-0")
                ], className="metric-card text-center")
            ], width=3),
            dbc.Col([
                html.Div([
                    html.H3(f"{basic_stats['total_roads']:,}", className="text-success mb-1"),
                    html.P("Total Roads", className="mb-0")
                ], className="metric-card text-center")
            ], width=3),
            dbc.Col([
                html.Div([
                    html.H3(f"{basic_stats['average_degree']}", className="text-warning mb-1"),
                    html.P("Average Degree", className="mb-0")
                ], className="metric-card text-center")
            ], width=3),
            dbc.Col([
                html.Div([
                    html.H3(f"{len(high_degree)}", className="text-info mb-1"),
                    html.P("High Degree Nodes", className="mb-0")
                ], className="metric-card text-center")
            ], width=3),
        ], className="mb-4"),
        
        # Network Health Indicators
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H5("Network Health Indicators", className="mb-3"),
                    dcc.Graph(figure=gauges_fig, config={'displayModeBar': False})
                ], className="chart-container")
            ], width=12)
        ], className="mb-4"),
        
        # Main Charts Row 1
        dbc.Row([
            dbc.Col([
                html.Div([
                    dcc.Graph(figure=degree_fig, config={'displayModeBar': False})
                ], className="chart-container")
            ], width=6),
            dbc.Col([
                html.Div([
                    dcc.Graph(figure=categories_fig, config={'displayModeBar': False})
                ], className="chart-container")
            ], width=6),
        ]),
        
        # Main Charts Row 2
        dbc.Row([
            dbc.Col([
                html.Div([
                    dcc.Graph(figure=top_connected_fig, config={'displayModeBar': False})
                ], className="chart-container")
            ], width=6),
            dbc.Col([
                html.Div([
                    dcc.Graph(figure=geo_fig, config={'displayModeBar': False})
                ], className="chart-container")
            ], width=6),
        ]),
        
        

        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H5("🚦 Shortest Path Finder", className="mb-3"),
                    dbc.InputGroup([
                        dbc.Input(id="source-id", placeholder="Source Intersection ID", type="number"),
                        dbc.Input(id="target-id", placeholder="Target Intersection ID", type="number"),
                        dbc.Button("Find Path", id="find-path-btn", color="primary")
                    ], className="mb-3"),
                    html.Div(id="shortest-path-output")
                ], className="chart-container")
            ], width=6),

            dbc.Col([
                html.Div([
                    html.H5("🔢 High Degree Intersections", className="mb-3"),
                    dbc.InputGroup([
                        dbc.Input(id="degree-threshold", placeholder="Degree Threshold", type="number", value=3),
                        dbc.Button("Query", id="query-degree-btn", color="secondary")
                    ], className="mb-3"),
                    html.Div(id="degree-query-output")
                ], className="chart-container")
            ], width=6),
        ]),


        # Analysis Insights
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H5("🔍 Key Insights", className="mb-3"),
                    html.Ul([
                        html.Li(f"The network contains {basic_stats['total_intersections']:,} intersections connected by {basic_stats['total_roads']:,} roads."),
                        html.Li(f"Average connectivity is {basic_stats['average_degree']} roads per intersection, indicating a sparse network."),
                        html.Li(f"Most intersections have low connectivity (≤2 connections), typical of road networks."),
                        html.Li("Geographic distribution shows clustering patterns that reflect urban vs rural areas."),
                        html.Li("High-degree intersections likely represent major highway interchanges or city centers.")
                    ], style={'fontSize': '1rem', 'lineHeight': '1.6'})
                ], className="chart-container")
            ], width=12)
        ]),

        
        # Footer
        html.Hr(),
        html.P("📊 MSDA9215: Big Data Analytics - Neo4j Road Network Analysis", 
               className="text-center text-muted")
        
    ], fluid=True, style={'backgroundColor': '#f8f9fa'})

    @app.callback(
        Output("shortest-path-output", "children"),
        Input("find-path-btn", "n_clicks"),
        State("source-id", "value"),
        State("target-id", "value")
    )
    def find_shortest_path(n_clicks, source_id, target_id):
        if not n_clicks or source_id is None or target_id is None:
            raise PreventUpdate
        try:
            result = analyzer.find_shortest_path(int(source_id), int(target_id))
            if not result['path_ids']:
                return html.P("No path found.", style={"color": "orange"})
            return html.Div([
                html.P(f"Path Length: {result['path_length']} hops"),
                html.P("Path: " + " → ".join(map(str, result['path_ids'])), style={"fontWeight": "bold"})
            ])
        except Exception as e:
            return html.Div([
                html.P("Error calculating path."),
                html.Pre(str(e))
            ], style={"color": "red"})

    @app.callback(
        Output("degree-query-output", "children"),
        Input("query-degree-btn", "n_clicks"),
        State("degree-threshold", "value")
    )
    def query_high_degree(n_clicks, threshold):
        if not n_clicks or threshold is None:
            raise PreventUpdate
        try:
            intersections = analyzer.get_high_degree_intersections(int(threshold))
            if not intersections:
                return html.P("No intersections found with degree greater than threshold.")

            return html.Div([
                html.P(f"Total results: {len(intersections)}"),
                html.Ul([
                    html.Li(f"ID {item['id']} - Degree {item['degree']}") for item in intersections
                ], style={"maxHeight": "200px", "overflowY": "scroll"})
            ])
        except Exception as e:
            return html.Div([
                html.P("Error querying intersections."),
                html.Pre(str(e))
            ], style={"color": "red"})
        

    @app.callback(
        Output("shortest-path-result", "children"),
        Input("btn-shortest-path", "n_clicks"),
        [Input("source-id", "value"), Input("target-id", "value")]
    )
    def update_shortest_path(n_clicks, source_id, target_id):
        if n_clicks is None or source_id is None or target_id is None:
            raise PreventUpdate
        try:
            result = analyzer.find_shortest_path(source_id, target_id)
            return f"Path length: {result['path_length']} nodes, Total distance: {result['total_distance']:.2f}"
        except Exception as e:
            return f"Error: {str(e)}"

    @app.callback(
        Output("degree-threshold-graph", "figure"),
        Input("btn-degree-query", "n_clicks"),
        Input("degree-threshold", "value")
    )
    def update_degree_threshold_graph(n_clicks, threshold):
        if n_clicks is None or threshold is None:
            raise PreventUpdate
        try:
            intersections = analyzer.get_high_degree_intersections(threshold)
            if not intersections:
                fig = go.Figure()
                fig.add_annotation(text="No intersections found", x=0.5, y=0.5, showarrow=False)
                return fig
            fig = px.scatter(
                intersections, x="x", y="y", size="degree", color="degree",
                hover_name="id", title=f"Intersections with Degree > {threshold}",
                color_continuous_scale="Inferno"
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            return fig
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Error: {str(e)}", x=0.5, y=0.5, showarrow=False)
            return fig
    
    return app

def main():
    """Main execution function"""
    print("🔗 Connecting to Neo4j database...")
    
    try:
        # Initialize analyzer
        analyzer = USRoadNetworkAnalyzer()
        
        # Load data from file
        print("📂 Loading data into Neo4j...")
        analyzer.load_data_from_file("usa.txt")
        
        # Perform analysis
        print("\n=== US ROAD NETWORK ANALYSIS RESULTS ===")
        
        # 1. Basic Statistics
        stats = analyzer.get_basic_stats()
        print(f"\n1. BASIC NETWORK STATISTICS:")
        print(f"   Total Intersections: {stats['total_intersections']:,}")
        print(f"   Total Roads: {stats['total_roads']:,}")
        print(f"   Average Degree: {stats['average_degree']}")
        
        # 2. Shortest Path Example
        if stats['total_intersections'] > 1:
            print(f"\n2. SHORTEST PATH ANALYSIS:")
            try:
                path_result = analyzer.find_shortest_path(0, 10)
                print(f"   Path from intersection 0 to 10:")
                print(f"   - Path length: {path_result['path_length']} nodes")
                print(f"   - Total distance: {path_result['total_distance']:.2f}")
            except Exception as e:
                print(f"   Shortest path calculation requires APOC plugin: {e}")
        
        # 3. High Degree Intersections
        high_degree = analyzer.get_high_degree_intersections(3)
        print(f"\n3. HIGH DEGREE INTERSECTIONS (>3 connections):")
        print(f"   Found {len(high_degree)} intersections")
        for intersection in high_degree[:5]:
            print(f"   - ID {intersection['id']}: {intersection['degree']} connections")
        
        # 4. Betweenness Centrality
        print(f"\n4. BETWEENNESS CENTRALITY ANALYSIS:")
        try:
            centrality = analyzer.calculate_betweenness_centrality(50)
            if centrality:
                centrality.sort(key=lambda x: x['centrality'], reverse=True)
                print(f"   Top 3 central nodes (sample of 50):")
                for node in centrality[:3]:
                    print(f"   - ID {node['id']}: {node['centrality']:.4f}")
            else:
                print("   No centrality data available")
        except Exception as e:
            print(f"   Centrality calculation skipped: {e}")
        
        # 5. Degree Distribution
        degree_dist = analyzer.get_degree_distribution()
        print(f"\n5. DEGREE DISTRIBUTION:")
        for degree, count in sorted(degree_dist.items())[:10]:
            print(f"   Degree {degree}: {count} intersections")
        
        # 6. Connectivity Categories
        categories = analyzer.categorize_intersections_by_degree()
        print(f"\n6. CONNECTIVITY CATEGORIES:")
        for category, count in categories.items():
            print(f"   {category}: {count} intersections")
        
        print(f"\n7. LAUNCHING INTERACTIVE DASHBOARD...")
        print(f"   Dashboard will be available at: http://127.0.0.1:8050")
        print(f"   Press Ctrl+C to stop the server")
        
        # Create and run dashboard
        app = create_dashboard(analyzer)
        app.run(debug=False, host='127.0.0.1', port=8050)
    
    except ConnectionError as e:
        print(f"❌ Connection Error: {e}")
        print("\n🔧 Troubleshooting steps:")
        print("1. Start Neo4j Desktop or Neo4j Server")
        print("2. Verify Neo4j is running on bolt://localhost:7687")
        print("3. Check username/password (default: neo4j/password)")
        print("4. Ensure firewall allows Neo4j connections")
        
    except FileNotFoundError:
        print("❌ File Error: 'usa.txt' not found")
        print("Please ensure the data file exists in the current directory")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        print(f"❌ Unexpected Error: {str(e)}")
        print("\n🔧 Please check:")
        print("1. All required Python packages are installed")
        print("2. Neo4j database is accessible")
        print("3. Data file format is correct")
    
    finally:
        try:
            analyzer.close()
        except:
            pass

if __name__ == "__main__":
    # Required packages installation notice
    required_packages = [
        "neo4j", "pandas", "numpy", "plotly", "dash", 
        "dash-bootstrap-components", "networkx"
    ]
    
    print("🚗 US Road Network Analysis with Neo4j")
    print("=" * 50)
    print(f"Required packages: {', '.join(required_packages)}")
    print("Install with: pip install " + " ".join(required_packages))
    print("=" * 50)
    
    main()