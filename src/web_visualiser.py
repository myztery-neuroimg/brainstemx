#!/usr/bin/env python3
"""
Dash UI v5 — three-plane browser + Freeview helper.

• Select subject, modality
• Toggle lesion / brain-stem overlays
• Three orthogonal sliders (x, y, z)
• Shows a bash command to open current subject in Freeview
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
    nii=nib.load(str(path)); return nii.get_fdata().astype(np.float32), nii.affine

def overlay(mask): return np.ma.masked_where(mask==0, mask)

def base_path(subj,mod):
    return {"flair":"flair_to_t1.nii.gz",
            "dwi":"dwi_to_t1.nii.gz",
            "swi":"swi_to_t1.nii.gz"}[mod]

# ---------- callbacks --------------------------------------------------------
@app.callback(
    Output("sl_x","max"),Output("sl_y","max"),Output("sl_z","max"),
    Output("view","figure"),Output("cmd","children"),
    Input("subj","value"),Input("mod","value"),
    Input("sl_x","value"),Input("sl_y","value"),Input("sl_z","value"),
    Input("ov","value"))
def update(subj,mod,x,y,z,ovs):
    if not subj: return 100,100,100,{}, ""
    p=ROOT/subj
    base,aff=load(p/base_path(subj,mod))
    x,y,z=[int(np.clip(v,0,base.shape[i]-1)) for i,v in enumerate((x,y,z))]
    fig=go.Figure()
    fig.add_trace(go.Image(z=base[:,:,z].T))
    if "bs" in ovs and (p/"brainstem_mask.nii.gz").exists():
        bs,_=load(p/"brainstem_mask.nii.gz")
        fig.add_trace(go.Image(z=overlay(bs[:,:,z].T), colorscale="Blues", opacity=0.3))
    if "les" in ovs and (p/"lesion_sd2.0.nii.gz").exists():
        les,_=load(p/"lesion_sd2.0.nii.gz")
        fig.add_trace(go.Image(z=overlay(les[:,:,z].T), colorscale="Reds", opacity=0.5))
    fig.update_layout(margin=dict(l=0,r=0,b=0,t=20))
    freeview_cmd = (
        f"freeview -v {p/base_path(subj,mod)} "
        f"{p/'lesion_sd2.0.nii.gz'}:colormap=heat:opacity=0.5 "
        f"{p/'brainstem_mask.nii.gz'}:colormap=blue")
    return base.shape[0]-1,base.shape[1]-1,base.shape[2]-1,fig,freeview_cmd

if __name__ == "__main__":
    import argparse; ap=argparse.ArgumentParser(); ap.add_argument("--root",default="results"); ap.add_argument("--port",type=int,default=8050)
    args=ap.parse_args(); ROOT=Path(args.root)
    app.run_server(debug=False, port=args.port)

