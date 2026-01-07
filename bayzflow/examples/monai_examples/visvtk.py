#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
import vtk


# ==============================
# 1. CONFIG: patient folders
# ==============================

PATIENT_A_DIR = Path("latent_dump/patient_0")
PATIENT_B_DIR = Path("latent_dump/patient_2")

# limit how many posterior trajectories to actually draw (for clarity)
MAX_TRAJS_PER_PATIENT = 10

# tube + sphere scale – adjust depending on PCA scale
TUBE_RADIUS_POSTERIOR = 0.6
TUBE_RADIUS_MEAN = 1.0
SPHERE_RADIUS = 1.8


# ==============================
# 2. Loading utilities
# ==============================

FNAME_RE = re.compile(r"sample_(\d+)_layer_(\d+)\.npy")


def load_patient_latents_from_folder(folder: Path):
    """
    Load per-sample, per-layer latent vectors from a folder with files like:
        sample_000_layer_00.npy

    Returns:
        Z_pad: [S, L, D_pad]  (zero-padded across all (sample, layer))
        layer_indices: sorted list of layer integers [0..L-1]
    """
    folder = Path(folder)
    files = sorted(
        [f for f in folder.glob("sample_*_layer_*.npy") if f.is_file()]
    )
    if not files:
        raise RuntimeError(f"No sample_XXX_layer_YY.npy files in {folder}")

    # Parse all (sample, layer) indices and shapes
    records = []
    max_sample = -1
    max_layer = -1
    max_dim = 0

    for f in files:
        m = FNAME_RE.match(f.name)
        if not m:
            continue
        s_idx = int(m.group(1))
        l_idx = int(m.group(2))

        v = np.load(f)
        if v.ndim != 1:
            v = v.reshape(-1)
        d = v.shape[0]

        records.append((s_idx, l_idx, d, f))
        max_sample = max(max_sample, s_idx)
        max_layer = max(max_layer, l_idx)
        max_dim = max(max_dim, d)

    S = max_sample + 1  # assuming contiguous 0..S-1
    L = max_layer + 1   # assuming contiguous 0..L-1
    D_pad = max_dim

    if S <= 0 or L <= 0:
        raise RuntimeError(f"Could not infer S/L from files in {folder}")

    # Allocate padded array
    Z = np.zeros((S, L, D_pad), dtype=np.float32)

    # Fill it
    for s_idx, l_idx, d, f in records:
        v = np.load(f).astype(np.float32)
        if v.ndim != 1:
            v = v.reshape(-1)
        Z[s_idx, l_idx, :d] = v

    # Return plus the layer indices for reference
    layer_indices = list(range(L))
    return Z, layer_indices


def build_joint_pca_from_two_patients(ZA, ZB):
    """
    ZA: [S_a, L, D_pad]
    ZB: [S_b, L, D_pad]

    Returns:
        XA_3d: [S_a, L, 3]
        XB_3d: [S_b, L, 3]
    """
    S_a, L_a, D = ZA.shape
    S_b, L_b, D2 = ZB.shape
    assert D == D2, "Patients must have same padded latent dim"
    if L_a != L_b:
        # just in case; clip to common layers
        L = min(L_a, L_b)
        ZA = ZA[:, :L, :]
        ZB = ZB[:, :L, :]
    else:
        L = L_a

    Z_all = np.concatenate(
        [ZA.reshape(S_a * L, D), ZB.reshape(S_b * L, D)],
        axis=0,
    )  # [S_a*L + S_b*L, D]

    pca = PCA(n_components=3)
    X_all = pca.fit_transform(Z_all)  # [total, 3]

    XA_3d = X_all[: S_a * L].reshape(S_a, L, 3)
    XB_3d = X_all[S_a * L :].reshape(S_b, L, 3)
    return XA_3d, XB_3d


# ==============================
# 3. VTK helpers
# ==============================

def make_tube_actor(points_np, color, radius=1.0, opacity=0.25):
    """
    Build a tube actor for a polyline through points_np: [N, 3]
    """
    points = vtk.vtkPoints()
    n = points_np.shape[0]
    for i in range(n):
        x, y, z = points_np[i]
        points.InsertNextPoint(float(x), float(y), float(z))

    lines = vtk.vtkCellArray()
    lines.InsertNextCell(n)
    for i in range(n):
        lines.InsertCellPoint(i)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)

    tube = vtk.vtkTubeFilter()
    tube.SetInputData(polydata)
    tube.SetRadius(radius)
    tube.SetNumberOfSides(20)
    tube.CappingOn()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(tube.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(opacity)

    return actor


def make_sphere_actor(center, radius, color):
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(radius)
    sphere.SetThetaResolution(32)
    sphere.SetPhiResolution(32)
    sphere.SetCenter(*center)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(sphere.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(1.0)

    return sphere, actor


# ==============================
# 4. Main visualisation setup
# ==============================

def main():
    # ---- load latents from folders ----
    print(f"Loading patient A from: {PATIENT_A_DIR}")
    ZA, layersA = load_patient_latents_from_folder(PATIENT_A_DIR)  # [S_a, L, D_pad]
    print(f"Loading patient B from: {PATIENT_B_DIR}")
    ZB, layersB = load_patient_latents_from_folder(PATIENT_B_DIR)  # [S_b, L, D_pad]

    print(f"Patient A: {ZA.shape[0]} samples, {ZA.shape[1]} layers")
    print(f"Patient B: {ZB.shape[0]} samples, {ZB.shape[1]} layers")

    # ---- PCA to 3D ----
    XA_3d, XB_3d = build_joint_pca_from_two_patients(ZA, ZB)
    S_a, L, _ = XA_3d.shape
    S_b, _, _ = XB_3d.shape

    # mean trajectories
    meanA = XA_3d.mean(axis=0)  # [L, 3]
    meanB = XB_3d.mean(axis=0)  # [L, 3]

    # ---- VTK renderer & window ----
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.02, 0.02, 0.06)

    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(1000, 800)
    render_window.AddRenderer(renderer)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # colours
    red = (1.0, 0.3, 0.3)
    blue = (0.3, 0.5, 1.0)

    # ---- posterior tubes (faint) ----
    for s in range(min(S_a, MAX_TRAJS_PER_PATIENT)):
        actor = make_tube_actor(
            XA_3d[s],
            red,
            radius=TUBE_RADIUS_POSTERIOR,
            opacity=0.15,
        )
        renderer.AddActor(actor)

    for s in range(min(S_b, MAX_TRAJS_PER_PATIENT)):
        actor = make_tube_actor(
            XB_3d[s],
            blue,
            radius=TUBE_RADIUS_POSTERIOR,
            opacity=0.15,
        )
        renderer.AddActor(actor)

    # ---- mean trajectory tubes (strong) ----
    meanA_actor = make_tube_actor(
        meanA,
        red,
        radius=TUBE_RADIUS_MEAN,
        opacity=0.95,
    )
    meanB_actor = make_tube_actor(
        meanB,
        blue,
        radius=TUBE_RADIUS_MEAN,
        opacity=0.95,
    )
    renderer.AddActor(meanA_actor)
    renderer.AddActor(meanB_actor)

    # ---- moving spheres to show belief evolution ----
    sphereA, sphereA_actor = make_sphere_actor(
        center=meanA[0],
        radius=SPHERE_RADIUS,
        color=(1.0, 0.7, 0.7),
    )
    sphereB, sphereB_actor = make_sphere_actor(
        center=meanB[0],
        radius=SPHERE_RADIUS,
        color=(0.7, 0.8, 1.0),
    )

    renderer.AddActor(sphereA_actor)
    renderer.AddActor(sphereB_actor)

    # camera
    renderer.ResetCamera()
    cam = renderer.GetActiveCamera()
    cam.Azimuth(35)
    cam.Elevation(20)
    renderer.ResetCameraClippingRange()

    # ---- animation callback ----
    class BeliefAnimationCallback:
        def __init__(self, meanA, meanB, sphereA, sphereB, renderer, window):
            self.meanA = meanA
            self.meanB = meanB
            self.sphereA = sphereA
            self.sphereB = sphereB
            self.renderer = renderer
            self.window = window
            self.t = 0
            self.L = meanA.shape[0]

        def execute(self, obj, event):
            layer = self.t % self.L
            xA, yA, zA = self.meanA[layer]
            xB, yB, zB = self.meanB[layer]

            self.sphereA.SetCenter(float(xA), float(yA), float(zA))
            self.sphereB.SetCenter(float(xB), float(yB), float(zB))

            cam = self.renderer.GetActiveCamera()
            cam.Azimuth(0.8)  # slow orbit
            self.renderer.ResetCameraClippingRange()

            self.window.Render()
            self.t += 1

    cb = BeliefAnimationCallback(meanA, meanB, sphereA, sphereB, renderer, render_window)
    interactor.AddObserver("TimerEvent", cb.execute)

    interactor.Initialize()
    interactor.CreateRepeatingTimer(120)  # ms per frame
    render_window.Render()
    interactor.Start()


if __name__ == "__main__":
    main()
