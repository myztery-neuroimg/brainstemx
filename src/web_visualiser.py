#!/usr/bin/env python3
"""
Dash UI v5 — three-plane browser + Freeview helper.

• Select subject, modality (FLAIR/DWI/SWI)
• Toggle lesion / brain-stem overlays
• Three orthogonal sliders (x, y, z)
• Shows a bash command to open current subject in Freeview
• Gracefully handles missing modalities
"""

import base64, tempfile
from pathlib import Path
import dash, dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State
import plotly.graph_objects as go
import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path("results")

def subj_opts():
    return [{"label":p.name,"value":p.name} for p in ROOT.iterdir() if (p/"summary.json").exists()]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H4("BrainStemX UI v5"),
            dcc.Dropdown(id="subj", options=subj_opts(), placeholder="select subject"),
            html.H6("Modality"),
            dcc.Dropdown(id="mod", options=[
                {"label":"FLAIR","value":"flair"},
                {"label":"DWI","value":"dwi"},
                {"label":"SWI","value":"swi"}], value="flair"),
            html.H6("Overlays"),
            dcc.Checklist(id="ov", options=[
                {"label":"Lesion (2 SD)","value":"les"},
                {"label":"Brain-stem mask","value":"bs"}],
                value=["les"]),
            html.Hr(),
            html.Pre(id="cmd", style={"fontSize":"0.8rem"})
        ], width=2),

        dbc.Col([
            dcc.Slider(id="sl_x", min=0,max=100,value=50,tooltip={"placement":"bottom"}),
            dcc.Slider(id="sl_y", min=0,max=100,value=50,tooltip={"placement":"bottom"}),
            dcc.Slider(id="sl_z", min=0,max=100,value=50,tooltip={"placement":"bottom"}),
            dcc.Graph(id="view"),
        ], width=10)
    ])
], fluid=True)

# ---------- helpers ----------------------------------------------------------
def load(path):
    """Load a NIfTI file and return its data and affine transform."""
    nii=nib.load(str(path))
    return nii.get_fdata().astype(np.float32), nii.affine

def overlay(mask):
    """Create a masked array for overlay visualization."""
    return np.ma.masked_where(mask==0, mask)

def get_available_modalities(subject_dir: Path) -> list:
    """Get list of available modalities for a subject."""
    modalities = []
    if (subject_dir/"flair_to_t1.nii.gz").exists():
        modalities.append("flair")
    if (subject_dir/"dwi_to_t1.nii.gz").exists():
        modalities.append("dwi")
    if (subject_dir/"swi_to_t1.nii.gz").exists():
        modalities.append("swi")
    return modalities

def base_path(subject_dir: Path, mod: str) -> Path:
    """Get the path for the requested modality, or None if not available."""
    path_map = {
        "flair": "flair_to_t1.nii.gz",
        "dwi": "dwi_to_t1.nii.gz",
        "swi": "swi_to_t1.nii.gz"
    }
    
    if mod not in path_map:
        return None
        
    file_path = subject_dir/path_map[mod]
    if not file_path.exists():
        return None
        
    return file_path

# ---------- callbacks --------------------------------------------------------
@app.callback(
    Output("sl_x","max"),Output("sl_y","max"),Output("sl_z","max"),
    Output("view","figure"),Output("cmd","children"),
    Input("subj","value"),Input("mod","value"),
    Input("sl_x","value"),Input("sl_y","value"),Input("sl_z","value"),
    Input("ov","value"))
def update(subj, mod, x, y, z, ovs):
    """Update visualization based on selected subject, modality and overlays."""
    if not subj:
        return 100, 100, 100, go.Figure(), ""
        
    p = ROOT/subj
    
    # Check if the requested modality is available
    file_path = base_path(p, mod)
    
    # If the requested modality isn't available, fall back to an available one
    if file_path is None:
        available_mods = get_available_modalities(p)
        if not available_mods:
            fig = go.Figure()
            fig.add_annotation(text="No imaging data available",
                              x=0.5, y=0.5, showarrow=False,
                              font=dict(size=20))
            return 100, 100, 100, fig, f"No modalities available for {subj}"
            
        # Fall back to first available modality
        mod = available_mods[0]
        file_path = p/base_path(p, mod)
        if file_path is None:  # Should never happen but just for safety
            return 100, 100, 100, go.Figure(), ""
    
    # Load the base image
    try:
        base, aff = load(file_path)
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error loading {mod} image: {str(e)}",
                          x=0.5, y=0.5, showarrow=False)
        return 100, 100, 100, fig, ""
    
    # Update slice indices based on image dimensions
    x, y, z = [int(np.clip(v, 0, base.shape[i]-1)) for i, v in enumerate((x, y, z))]
    
    # Create figure
    fig = go.Figure()
    fig.add_trace(go.Image(z=base[:,:,z].T))
    
    # Add overlays if available
    if "bs" in ovs and (p/"brainstem_mask.nii.gz").exists():
        bs, _ = load(p/"brainstem_mask.nii.gz")
        fig.add_trace(go.Image(z=overlay(bs[:,:,z].T), colorscale="Blues", opacity=0.3))
    
    if "les" in ovs and (p/"lesion_sd2.0.nii.gz").exists():
        les, _ = load(p/"lesion_sd2.0.nii.gz")
        fig.add_trace(go.Image(z=overlay(les[:,:,z].T), colorscale="Reds", opacity=0.5))
    
    # Layout adjustments
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=20))
    
    # Generate freeview command
    overlay_options = []
    
    if (p/"lesion_sd2.0.nii.gz").exists():
        overlay_options.append(f"{p/'lesion_sd2.0.nii.gz'}:colormap=heat:opacity=0.5")
        
    if (p/"brainstem_mask.nii.gz").exists():
        overlay_options.append(f"{p/'brainstem_mask.nii.gz'}:colormap=blue")
        
    overlay_str = " ".join(overlay_options)
    freeview_cmd = f"freeview -v {file_path} {overlay_str}"
    
    return base.shape[0]-1, base.shape[1]-1, base.shape[2]-1, fig, freeview_cmd

def main():
    """Run the Dash visualization server."""
    import argparse
    ap = argparse.ArgumentParser(description="BrainStem X Web Visualizer")
    ap.add_argument("--root", default="results", help="Root directory containing subject folders")
    ap.add_argument("--port", type=int, default=8050, help="Port for the web server")
    ap.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = ap.parse_args()
    global ROOT
    ROOT = Path(args.root)
    
    print(f"Starting BrainStem X visualizer on http://localhost:{args.port}")
    print(f"Root directory: {ROOT}")
    
    app.run_server(debug=args.debug, port=args.port)

if __name__ == "__main__":
    main()

