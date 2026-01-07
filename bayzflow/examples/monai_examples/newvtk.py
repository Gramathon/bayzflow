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

# === NEW: optional volume .npy paths for the two patients ===
# These should be 3D arrays shaped (z, y, x). Replace with your real paths.
VOLUME_A_NPY = Path("volumes/patient_0_volume.npy")
VOLUME_B_NPY = Path("volumes/patient_2_volume.npy")

# limit how many posterior trajectories to actually draw (for clarity)
MAX_TRAJS_PER_PATIENT = 20

# tube + sphere scale – adjust depending on PCA scale
TUBE_RADIUS_POSTERIOR = 0.25
TUBE_RADIUS_MEAN = 0.6
SPHERE_RADIUS = 1.0


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
    tube.SetNumberOfSides(32)
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


def make_point_cloud_actor(points_np, color, point_radius=0.25, opacity=0.25):
    """
    Draw small spheres at each point in points_np: [N, 3],
    using vtkGlyph3D for efficiency.
    """
    points = vtk.vtkPoints()
    n = points_np.shape[0]
    for i in range(n):
        x, y, z = points_np[i]
        points.InsertNextPoint(float(x), float(y), float(z))

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)

    # Sphere glyph
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(point_radius)
    sphere.SetThetaResolution(12)
    sphere.SetPhiResolution(12)

    glyph = vtk.vtkGlyph3D()
    glyph.SetInputData(polydata)
    glyph.SetSourceConnection(sphere.GetOutputPort())
    glyph.ScalingOff()
    glyph.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(glyph.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(opacity)

    return actor

def add_outline_for_volume(renderer, volume_shape):
    """
    Draw a yellow wireframe box matching the volume extents
    so you can at least see *something* in the viewport.
    volume_shape: (z, y, x)
    """
    z, y, x = volume_shape

    img = vtk.vtkImageData()
    img.SetDimensions(x, y, z)

    outline = vtk.vtkOutlineFilter()
    outline.SetInputData(img)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(outline.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.9, 0.9, 0.1)
    actor.GetProperty().SetLineWidth(2.0)

    renderer.AddActor(actor)



# === NEW: volume helpers (NumPy 3D -> vtkVolume) ===================
def numpy_volume_to_vtk_image(volume: np.ndarray,
                              spacing=(1.0, 1.0, 1.0)) -> vtk.vtkImageData:
    """
    Convert a 3D numpy array (z, y, x) to vtkImageData.
    """
    vol = np.ascontiguousarray(volume)
    z, y, x = vol.shape

    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(x, y, z)
    vtk_image.SetSpacing(spacing[0], spacing[1], spacing[2])
    vtk_image.AllocateScalars(vtk.VTK_FLOAT, 1)

    for kz in range(z):
        for ky in range(y):
            for kx in range(x):
                vtk_image.SetScalarComponentFromFloat(
                    kx, ky, kz, 0, float(vol[kz, ky, kx])
                )

    return vtk_image


def make_volume_actor(volume: np.ndarray,
                      spacing=(1.0, 1.0, 1.0),
                      opacity=0.2) -> vtk.vtkVolume:
    """
    Create a vtkVolume actor from a numpy 3D array.
    Prints min/max so you can debug intensity.
    """
    print("Volume shape:", volume.shape,
          "dtype:", volume.dtype,
          "min:", float(volume.min()),
          "max:", float(volume.max()))

    image = numpy_volume_to_vtk_image(volume, spacing=spacing)

    vmin, vmax = float(volume.min()), float(volume.max())
    if vmin == vmax:
        # Avoid degenerate transfer function
        vmax = vmin + 1.0

    mid = (vmin + vmax) / 2.0

    opacity_tf = vtk.vtkPiecewiseFunction()
    # Be quite aggressive so you definitely see something
    opacity_tf.AddPoint(vmin, 0.0)
    opacity_tf.AddPoint(mid, opacity)
    opacity_tf.AddPoint(vmax, 0.9)

    color_tf = vtk.vtkColorTransferFunction()
    color_tf.AddRGBPoint(vmin, 0.0, 0.0, 0.0)
    color_tf.AddRGBPoint(vmax, 1.0, 1.0, 1.0)

    volume_property = vtk.vtkVolumeProperty()
    volume_property.SetColor(color_tf)
    volume_property.SetScalarOpacity(opacity_tf)
    volume_property.SetInterpolationTypeToLinear()
    volume_property.ShadeOff()

    mapper = vtk.vtkSmartVolumeMapper()
    mapper.SetInputData(image)

    vol_actor = vtk.vtkVolume()
    vol_actor.SetMapper(mapper)
    vol_actor.SetProperty(volume_property)

    return vol_actor


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

    # pick "hero" posterior trajectories (e.g. first sample)
    heroA = XA_3d[0]  # [L, 3]
    heroB = XB_3d[0]

    # flatten all points for dense clouds
    ptsA = XA_3d.reshape(-1, 3)
    ptsB = XB_3d.reshape(-1, 3)

    # =============================
    # VTK: one window, 3 viewports
    # =============================

    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(1600, 900)

    # --- latent renderer (left half, full height) ---
    latent_renderer = vtk.vtkRenderer()
    latent_renderer.SetViewport(0.0, 0.0, 0.5, 1.0)
    latent_renderer.SetBackground(0.02, 0.02, 0.06)
    render_window.AddRenderer(latent_renderer)

    # --- volume 1 renderer (top-right) ---
    vol1_renderer = vtk.vtkRenderer()
    vol1_renderer.SetViewport(0.5, 0.5, 1.0, 1.0)
    vol1_renderer.SetBackground(0.02, 0.02, 0.02)
    render_window.AddRenderer(vol1_renderer)

    # --- volume 2 renderer (bottom-right) ---
    vol2_renderer = vtk.vtkRenderer()
    vol2_renderer.SetViewport(0.5, 0.0, 1.0, 0.5)
    vol2_renderer.SetBackground(0.02, 0.02, 0.02)
    render_window.AddRenderer(vol2_renderer)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # colours
    red = (1.0, 0.3, 0.3)
    blue = (0.3, 0.5, 1.0)

    # ---- dense point clouds for both patients ----
    cloudA_actor = make_point_cloud_actor(
        ptsA,
        color=red,
        point_radius=0.22,
        opacity=0.25,
    )
    cloudB_actor = make_point_cloud_actor(
        ptsB,
        color=blue,
        point_radius=0.22,
        opacity=0.25,
    )
    latent_renderer.AddActor(cloudA_actor)
    latent_renderer.AddActor(cloudB_actor)

    # ---- posterior tubes (faint bundle) ----
    for s in range(min(S_a, MAX_TRAJS_PER_PATIENT)):
        actor = make_tube_actor(
            XA_3d[s],
            red,
            radius=TUBE_RADIUS_POSTERIOR,
            opacity=0.12,
        )
        latent_renderer.AddActor(actor)

    for s in range(min(S_b, MAX_TRAJS_PER_PATIENT)):
        actor = make_tube_actor(
            XB_3d[s],
            blue,
            radius=TUBE_RADIUS_POSTERIOR,
            opacity=0.12,
        )
        latent_renderer.AddActor(actor)

    # ---- hero trajectories (bold) ----
    heroA_actor = make_tube_actor(
        heroA,
        red,
        radius=TUBE_RADIUS_MEAN * 1.2,
        opacity=0.95,
    )
    heroB_actor = make_tube_actor(
        heroB,
        blue,
        radius=TUBE_RADIUS_MEAN * 1.2,
        opacity=0.95,
    )
    latent_renderer.AddActor(heroA_actor)
    latent_renderer.AddActor(heroB_actor)

    # (optional) mean trajectories (slightly fainter)
    meanA_actor = make_tube_actor(
        meanA,
        red,
        radius=TUBE_RADIUS_MEAN,
        opacity=0.6,
    )
    meanB_actor = make_tube_actor(
        meanB,
        blue,
        radius=TUBE_RADIUS_MEAN,
        opacity=0.6,
    )
    latent_renderer.AddActor(meanA_actor)
    latent_renderer.AddActor(meanB_actor)

    # ---- moving spheres to show belief evolution along hero paths ----
    sphereA, sphereA_actor = make_sphere_actor(
        center=heroA[0],
        radius=SPHERE_RADIUS,
        color=(1.0, 0.7, 0.7),
    )
    sphereB, sphereB_actor = make_sphere_actor(
        center=heroB[0],
        radius=SPHERE_RADIUS,
        color=(0.7, 0.8, 1.0),
    )

    latent_renderer.AddActor(sphereA_actor)
    latent_renderer.AddActor(sphereB_actor)

    # ---- volumes (if available) ----
        # ---- volumes (if available) ----
    # Volume A
    if VOLUME_A_NPY.is_file():
        volA = np.load(VOLUME_A_NPY)
        print("[INFO] Loaded Volume A", volA.shape, volA.dtype)
    else:
        print(f"[WARN] Volume A npy not found at {VOLUME_A_NPY}, using dummy blob.")
        z, y, x = 64, 64, 64
        zz, yy, xx = np.meshgrid(
            np.linspace(-1, 1, z),
            np.linspace(-1, 1, y),
            np.linspace(-1, 1, x),
            indexing="ij",
        )
        volA = np.exp(-(xx**2 + yy**2 + zz**2) * 6.0)

    volume_actor1 = make_volume_actor(volA, spacing=(1.0, 1.0, 1.0), opacity=0.25)
    vol1_renderer.AddVolume(volume_actor1)
    add_outline_for_volume(vol1_renderer, volA.shape)
    vol1_renderer.ResetCamera()
    vol1_renderer.ResetCameraClippingRange()
    print("[INFO] Volume A added to top-right viewport.")

    # Volume B
    if VOLUME_B_NPY.is_file():
        volB = np.load(VOLUME_B_NPY)
        print("[INFO] Loaded Volume B", volB.shape, volB.dtype)
    else:
        print(f"[WARN] Volume B npy not found at {VOLUME_B_NPY}, using dummy shifted blob.")
        z, y, x = 64, 64, 64
        zz, yy, xx = np.meshgrid(
            np.linspace(-1, 1, z),
            np.linspace(-1, 1, y),
            np.linspace(-1, 1, x),
            indexing="ij",
        )
        volB = np.exp(-((xx + 0.3)**2 + (yy - 0.2)**2 + zz**2) * 8.0)

    volume_actor2 = make_volume_actor(volB, spacing=(1.0, 1.0, 1.0), opacity=0.25)
    vol2_renderer.AddVolume(volume_actor2)
    add_outline_for_volume(vol2_renderer, volB.shape)
    vol2_renderer.ResetCamera()
    vol2_renderer.ResetCameraClippingRange()
    print("[INFO] Volume B added to bottom-right viewport.")


    # camera for latent view
    latent_renderer.ResetCamera()
    cam = latent_renderer.GetActiveCamera()
    cam.Azimuth(35)
    cam.Elevation(20)
    latent_renderer.ResetCameraClippingRange()

    # ---- animation callback ----
    class BeliefAnimationCallback:
        def __init__(self, heroA, heroB, sphereA, sphereB,
                     latent_renderer, window):
            self.heroA = heroA
            self.heroB = heroB
            self.sphereA = sphereA
            self.sphereB = sphereB
            self.latent_renderer = latent_renderer
            self.window = window
            self.t = 0
            self.L = heroA.shape[0]

        def execute(self, obj, event):
            layer = self.t % self.L
            xA, yA, zA = self.heroA[layer]
            xB, yB, zB = self.heroB[layer]

            self.sphereA.SetCenter(float(xA), float(yA), float(zA))
            self.sphereB.SetCenter(float(xB), float(yB), float(zB))

            cam = self.latent_renderer.GetActiveCamera()
            cam.Azimuth(0.8)  # slow orbit
            self.latent_renderer.ResetCameraClippingRange()

            self.window.Render()
            self.t += 1

    cb = BeliefAnimationCallback(
        heroA, heroB, sphereA, sphereB, latent_renderer, render_window
    )
    interactor.AddObserver("TimerEvent", cb.execute)

    interactor.Initialize()
    interactor.CreateRepeatingTimer(120)  # ms per frame
    render_window.Render()
    interactor.Start()


if __name__ == "__main__":
    main()
