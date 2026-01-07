#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import torch
import numpy as np
import pyro

try:
    import vtk
except ImportError:
    print("You need to `pip install vtk` to run the 3D viewer.")
    sys.exit(1)


POSTERIOR_PATH = "bayz_artifacts/posterior_params.pt"


# -------------------------------------------------
# Load the param store
# -------------------------------------------------
def load_param_store(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Posterior file not found at: {path}")

    pyro.get_param_store().load(path)
    store = pyro.get_param_store()
    return store


# -------------------------------------------------
# Pretty-print the store contents
# -------------------------------------------------
def print_store_summary(store):
    print("\n=== Pyro Param Store Summary ===")
    for name, value in store.items():
        if isinstance(value, torch.Tensor):
            v = value.detach().cpu()
            shape_str = str(tuple(v.shape))  # SAFE!
            print(
                f"{name:30s} shape={shape_str:20s} "
                f"mean={v.mean().item():+.4f} std={v.std().item():+.4f}"
            )
        else:
            print(f"{name:30s} type={type(value)}")


# -------------------------------------------------
# Extract posterior mean & std vectors from the store
# -------------------------------------------------
def extract_mu_sigma(store):
    """
    Extract posterior mean/std for the final fc weight layer as a 1D latent vector.
    """
    # Try to grab the full fc weight posterior first
    if "AutoNormal.locs.fc.weight" in store and "AutoNormal.scales.fc.weight" in store:
        loc_name = "AutoNormal.locs.fc.weight"
        scale_name = "AutoNormal.scales.fc.weight"
        mu = store[loc_name].detach().cpu().numpy().reshape(-1)      # (10 * 32768,)
        sigma = store[scale_name].detach().cpu().numpy().reshape(-1) # (10 * 32768,)
        print(f"\nUsing loc='{loc_name}', scale='{scale_name}' (flattened)")
        print(f"Latent dimension D = {mu.shape[0]}")
        return mu, sigma

    # Fallback: previous heuristic (e.g. bias)
    loc_name = None
    scale_name = None
    for name, v in store.items():
        if not isinstance(v, torch.Tensor):
            continue
        if "loc" in name and v.ndim == 1:
            loc_name = name
        if ("scale" in name or "std" in name) and v.ndim == 1:
            scale_name = name

    if loc_name is None or scale_name is None:
        raise RuntimeError(
            "Could not find suitable loc/scale vectors. "
            "Use print_store_summary() to inspect available keys."
        )

    mu = store[loc_name].detach().cpu().numpy()
    sigma = store[scale_name].detach().cpu().numpy()
    print(f"\nUsing fallback loc='{loc_name}', scale='{scale_name}'")
    print(f"Latent dimension D = {mu.shape[0]}")
    return mu, sigma


# -------------------------------------------------
# Convert posterior (mu, sigma) to a VTK point cloud
# -------------------------------------------------
def build_vtk_point_cloud(mu: np.ndarray, sigma: np.ndarray, max_points: int = 50000):
    assert mu.shape == sigma.shape
    D = mu.shape[0]

    # Optional: random subsample for speed/clarity
    if D > max_points:
        idx = np.random.choice(D, size=max_points, replace=False)
        mu = mu[idx]
        sigma = sigma[idx]
        D = max_points

    # Normalise features
    mu_z = (mu - mu.mean()) / (mu.std() + 1e-8)
    log_sigma = np.log(sigma + 1e-8)
    log_sigma_z = (log_sigma - log_sigma.mean()) / (log_sigma.std() + 1e-8)

    idx_norm = np.linspace(0.0, 1.0, D, dtype=np.float32)

    points = vtk.vtkPoints()
    verts = vtk.vtkCellArray()
    scalars = vtk.vtkFloatArray()
    scalars.SetName("log_sigma_z")

    for i in range(D):
        x = float(mu_z[i])
        y = float(log_sigma_z[i])
        z = float(idx_norm[i])
        pid = points.InsertNextPoint(x, y, z)
        verts.InsertNextCell(1)
        verts.InsertCellPoint(pid)
        scalars.InsertNextValue(log_sigma_z[i])

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetVerts(verts)
    polydata.GetPointData().SetScalars(scalars)

    print(f"[VTK] Built point cloud with {points.GetNumberOfPoints()} points")
    return polydata


# -------------------------------------------------
# VTK renderer setup
# -------------------------------------------------
def make_vtk_pipeline(polydata):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.SetColorModeToMapScalars()
    mapper.ScalarVisibilityOn()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(4)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(0.07, 0.07, 0.10)

    ren_win = vtk.vtkRenderWindow()
    ren_win.AddRenderer(renderer)
    ren_win.SetSize(1200, 800)
    ren_win.SetWindowName("BayzFlow Posterior Latent State")

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(ren_win)

    renderer.ResetCamera()
    cam = renderer.GetActiveCamera()
    cam.Azimuth(30)
    cam.Elevation(25)
    renderer.ResetCameraClippingRange()

    return ren_win, iren


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    print(f"Loading posterior from: {POSTERIOR_PATH}")
    store = load_param_store(POSTERIOR_PATH)

    print_store_summary(store)

    mu, sigma = extract_mu_sigma(store)

    polydata = build_vtk_point_cloud(mu, sigma)

    ren_win, iren = make_vtk_pipeline(polydata)
    ren_win.Render()
    iren.Initialize()
    iren.Start()


if __name__ == "__main__":
    main()
