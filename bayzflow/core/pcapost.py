#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import torch
import pyro
from sklearn.decomposition import PCA

try:
    import vtk
except ImportError:
    print("pip install vtk")
    sys.exit(1)

POSTERIOR_PATH = "bayz_artifacts/posterior_params.pt"

# ---------------------  LOAD POSTERIOR  ---------------------
def load_store():
    pyro.get_param_store().load(POSTERIOR_PATH)
    return pyro.get_param_store()

# ---------------------  SAMPLE POSTERIOR  ---------------------
def sample_weight_vectors(store, num_samples=400):
    """
    Draw samples from the variational posterior for fc.weight.

    Returns: samples  [num_samples, D]  as numpy array.
    """
    # Grab guide params as TENSORS (no .cpu() yet)
    loc_t = store["AutoNormal.locs.fc.weight"].reshape(-1)
    scale_t = store["AutoNormal.scales.fc.weight"].reshape(-1)

    device = loc_t.device
    D = loc_t.shape[0]
    print(f"[posterior] latent dim = {D} on device {device}")

    samples = np.zeros((num_samples, D), dtype=np.float32)

    for i in range(num_samples):
        eps = torch.randn_like(loc_t)            # same device as loc_t/scale_t
        w = loc_t + scale_t * eps               # sample from q(w)
        samples[i] = w.detach().cpu().numpy()   # move to CPU for numpy/PCA

    print(f"[posterior] Generated {num_samples} samples")
    return samples

# ---------------------  PCA → 3D ---------------------
def pca_reduce(samples):
    pca = PCA(n_components=3)
    pts3d = pca.fit_transform(samples)
    print("[PCA] explained variance:", pca.explained_variance_ratio_)
    return pts3d

# ---------------------  VTK POINT CLOUD ---------------------
def vtk_cloud(points3d):
    points = vtk.vtkPoints()
    verts  = vtk.vtkCellArray()

    for i in range(points3d.shape[0]):
        x,y,z = points3d[i]
        pid = points.InsertNextPoint(float(x), float(y), float(z))
        verts.InsertNextCell(1)
        verts.InsertCellPoint(pid)

    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetVerts(verts)
    return poly

def render(poly):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(4)
    actor.GetProperty().SetColor(0.9, 0.6, 0.2)

    ren = vtk.vtkRenderer()
    ren.AddActor(actor)
    ren.SetBackground(0.05, 0.05, 0.08)

    win = vtk.vtkRenderWindow()
    win.AddRenderer(ren)
    win.SetSize(1200, 800)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(win)

    win.Render()
    iren.Initialize()
    iren.Start()

# ---------------------  MAIN ---------------------
def main():
    store = load_store()
    samples = sample_weight_vectors(store, num_samples=400)
    pts3d = pca_reduce(samples)
    poly = vtk_cloud(pts3d)
    render(poly)

if __name__ == "__main__":
    main()
