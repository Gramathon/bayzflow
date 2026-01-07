#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bayz Cosmic Belief Observatory

Loads multiple posterior snapshots:
    bayz_artifacts/trajectory/posterior_t0000.pt
    bayz_artifacts/trajectory/posterior_t0001.pt
    ...

For each snapshot:
  - samples K weight vectors from q(w | snapshot_t)
  - flattens them -> R^D

Then:
  - stacks all samples across time -> (T*K, D)
  - z-scores and does a single PCA -> (T*K, 3)
  - reshapes to clouds_3d[t, k, 3]

VTK:
  - keeps K points, and in a timer callback
    morphs them smoothly from cloud_t to cloud_{t+1}:
    - linear interpolation
    - quartile-based colours
    - shimmer & jitter
    - inertial camera tracking
    - evolving blob (belief field)
    - KNN link-graph between points
    - comet trail of belief centre
"""

import os
import re
import glob
import sys

import numpy as np
import torch
import pyro
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

try:
    import vtk
except ImportError:
    print("You need `pip install vtk` to run this viewer.")
    sys.exit(1)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
SNAP_DIR = "bayz_artifacts/trajectory"
SNAP_PATTERN = os.path.join(SNAP_DIR, "posterior_t*.pt")

LOC_KEY = "AutoNormal.locs.fc.weight"
SCALE_KEY = "AutoNormal.scales.fc.weight"

NUM_SAMPLES_PER_T = 600     # K points per cloud
JITTER_SCALE = 0.03         # spatial jitter scale in 3D latent space
TIMER_MS = 60               # timer interval for morph


# ---------------------------------------------------------------------
# Snapshot + posterior sampling helpers
# ---------------------------------------------------------------------
def find_snapshot_paths(pattern: str):
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"No posterior snapshots matched: {pattern}")

    def extract_t(path):
        m = re.search(r"t(\d+)\.pt$", os.path.basename(path))
        return int(m.group(1)) if m else 0

    paths.sort(key=extract_t)
    print(f"[snapshots] Found {len(paths)} snapshots.")
    for p in paths[:5]:
        print("  ", p)
    if len(paths) > 5:
        print("  ...")
    return paths


def sample_cloud_for_snapshot(path: str,
                              num_samples: int = NUM_SAMPLES_PER_T) -> np.ndarray:
    """
    For one posterior snapshot file:
      - load param_store
      - sample K weight vectors from Normal(loc, scale)
      - flatten each -> D
    Returns: cloud [K, D] as numpy.float32
    """
    pyro.get_param_store().load(path)
    store = pyro.get_param_store()

    if LOC_KEY not in store or SCALE_KEY not in store:
        raise KeyError(
            f"Expected loc/scale keys '{LOC_KEY}' and '{SCALE_KEY}' not found in {path}.\n"
            f"Available keys: {list(store.keys())}"
        )

    loc_t = store[LOC_KEY].reshape(-1)
    scale_t = store[SCALE_KEY].reshape(-1)
    device = loc_t.device
    D = loc_t.shape[0]

    print(f"[posterior] {os.path.basename(path)}  D={D}, device={device}")

    cloud = np.zeros((num_samples, D), dtype=np.float32)
    for i in range(num_samples):
        eps = torch.randn_like(loc_t)
        w = loc_t + scale_t * eps     # sample from q(w)
        cloud[i] = w.detach().cpu().numpy()

    return cloud


def build_all_clouds(paths, num_samples_per_t: int = NUM_SAMPLES_PER_T):
    """
    For each snapshot path, build one [K, D] cloud.
    Stack -> [T, K, D]
    """
    clouds = []
    for i, p in enumerate(paths):
        cloud = sample_cloud_for_snapshot(p, num_samples_per_t)
        clouds.append(cloud)
    clouds = np.stack(clouds, axis=0)  # [T, K, D]
    T, K, D = clouds.shape
    print(f"[clouds] shape = (T={T}, K={K}, D={D})")
    return clouds


# ---------------------------------------------------------------------
# PCA over all samples (global latent basis)
# ---------------------------------------------------------------------
def pca_clouds(clouds: np.ndarray):
    """
    clouds: [T, K, D]

    Flatten across time & samples -> (T*K, D),
    z-score per dimension, PCA -> 3D, then reshape back to [T, K, 3].
    """
    T, K, D = clouds.shape
    X = clouds.reshape(T * K, D)

    print("[Z-SCORE] Normalising weight samples across D...")
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True) + 1e-8
    X_z = (X - X_mean) / X_std

    print("[PCA] Fitting PCA on z-scored samples...")
    pca = PCA(n_components=3)
    X3 = pca.fit_transform(X_z)
    print("[PCA] explained variance ratio:", pca.explained_variance_ratio_)

    clouds3 = X3.reshape(T, K, 3)
    print("[PCA] clouds3 shape:", clouds3.shape)  # [T, K, 3]
    return clouds3, pca


# ---------------------------------------------------------------------
# VTK helpers: LUT, blob, KNN links, starfield
# ---------------------------------------------------------------------
def build_cosmic_lut():
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(4)
    lut.Build()
    lut.SetTableValue(0, 0.1, 0.4, 0.9, 1.0)   # inner: blue
    lut.SetTableValue(1, 0.0, 0.8, 0.9, 1.0)   # mid: cyan
    lut.SetTableValue(2, 0.9, 0.9, 0.2, 1.0)   # outer: yellow
    lut.SetTableValue(3, 0.95, 0.3, 0.9, 1.0)  # far: magenta
    return lut


def make_belief_blob(points3, radius=0.12, iso=0.25, dims=50):
    """
    Build a translucent blob around the cloud using:
    - Gaussian kernel point density
    - Marching cubes iso-surface
    """
    vtk_points = vtk.vtkPoints()
    for p in points3:
        vtk_points.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))

    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_points)

    kernel = vtk.vtkGaussianKernel()
    kernel.SetRadius(radius)
    kernel.SetSharpness(2.2)

    density = vtk.vtkPointDensityFilter()
    density.SetInputData(poly)
    # density.SetKernel(kernel)  # only if your VTK version supports it
    density.SetSampleDimensions(dims, dims, dims)
    # density.SetDensityEstimateToRelative()
    density.Update()

    contour = vtk.vtkMarchingCubes()
    contour.SetInputConnection(density.GetOutputPort())
    contour.SetValue(0, iso)
    contour.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(contour.GetOutputPort())
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(0.18)
    actor.GetProperty().SetColor(0.4, 0.7, 1.0)
    actor.GetProperty().SetSpecular(0.4)
    actor.GetProperty().SetSpecularPower(20)
    actor.GetProperty().SetInterpolationToPhong()

    return actor


def build_knn_edges(points3, k=6):
    """
    Build K-nearest-neighbour edges between points for visualising
    belief cohesion / tension structure.
    """
    nbr = NearestNeighbors(n_neighbors=k+1).fit(points3)
    dists, idxs = nbr.kneighbors(points3)

    lines = vtk.vtkCellArray()
    pts = vtk.vtkPoints()
    for p in points3:
        pts.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))

    for i in range(points3.shape[0]):
        for j in idxs[i][1:]:
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, i)
            line.GetPointIds().SetId(1, int(j))
            lines.InsertNextCell(line)

    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetLines(lines)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.2, 0.8, 1.0)
    actor.GetProperty().SetOpacity(0.15)
    actor.GetProperty().SetLineWidth(1.4)

    return actor


def build_starfield_actor(num_stars=800, radius=18.0):
    """
    Build a simple distant starfield: random points on a big sphere
    with tiny point size and low opacity.
    """
    pts = vtk.vtkPoints()
    verts = vtk.vtkCellArray()

    for _ in range(num_stars):
        # random point on sphere
        v = np.random.normal(size=3)
        v = v / (np.linalg.norm(v) + 1e-8)
        v = v * radius
        pid = pts.InsertNextPoint(float(v[0]), float(v[1]), float(v[2]))
        verts.InsertNextCell(1)
        verts.InsertCellPoint(pid)

    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetVerts(verts)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1.0, 1.0, 1.0)
    actor.GetProperty().SetPointSize(1.0)
    actor.GetProperty().SetOpacity(0.35)

    return actor


# ---------------------------------------------------------------------
# VTK: base morph callback (points only)
# ---------------------------------------------------------------------
class CosmicCloudMorphCallback:
    """
    Timer callback that:
      - interpolates between clouds[t] and clouds[t+1] in 3D
      - recomputes quartiles and colours
      - adds small jitter
      - shimmers opacity
      - camera tracks the cloud centre with inertia
    """

    def __init__(self, points, scalars, poly, clouds3, renderer):
        self.points = points
        self.scalars = scalars
        self.poly = poly
        self.clouds3 = clouds3         # [T, K, 3]
        self.renderer = renderer

        self.T, self.K, _ = clouds3.shape
        self.t_float = 0.0
        self.phase = 0.0

        # --- camera inertia state ---
        self.cam_init = False
        self.smoothed_center = None      # EMA over cloud centre
        self.cam_dir = None              # fixed direction from centre to camera
        self.cam_dist = None             # fixed distance from centre to camera

    # ---- latent-space time stepping ----
    def step_positions(self):
        # advance time (fractional index)
        self.t_float += 0.02            # speed of morphing
        self.phase += 0.07              # for shimmer

        base_t = int(self.t_float) % self.T
        next_t = (base_t + 1) % self.T
        frac = self.t_float - int(self.t_float)
        frac = max(0.0, min(1.0, frac))

        cloud_a = self.clouds3[base_t]      # [K, 3]
        cloud_b = self.clouds3[next_t]      # [K, 3]

        # linear interpolation between clouds
        pos = (1.0 - frac) * cloud_a + frac * cloud_b  # [K, 3]

        # jitter (scaled by spread)
        spread = np.linalg.norm(pos - pos.mean(axis=0, keepdims=True), axis=1).std() + 1e-8
        jitter = np.random.normal(scale=JITTER_SCALE * spread, size=pos.shape)
        pos_j = pos + jitter

        return pos_j

    # ---- colours & quartiles ----
    def update_points_and_colors(self, pos_j):
        centre = pos_j.mean(axis=0, keepdims=True)
        diffs = pos_j - centre
        dists = np.linalg.norm(diffs, axis=1)

        q1, q2, q3 = np.quantile(dists, [0.25, 0.5, 0.75])
        quart = np.zeros(self.K, dtype=np.int32)
        quart[dists > q1] = 1
        quart[dists > q2] = 2
        quart[dists > q3] = 3

        for i in range(self.K):
            x, y, z = pos_j[i]
            self.points.SetPoint(i, float(x), float(y), float(z))
            self.scalars.SetValue(i, int(quart[i]))

        self.points.Modified()
        self.scalars.Modified()
        self.poly.Modified()

    # ---- cinematic camera inertia ----
    def track_camera_inertial(self, centre_now, alpha_center=0.08, roll_strength=0.03):
        """
        centre_now: np.array shape (3,)
        alpha_center: how quickly camera focal point chases the cloud centre
        roll_strength: small roll around viewing direction for cinematic banking
        """
        cam = self.renderer.GetActiveCamera()

        centre_now = np.asarray(centre_now, dtype=np.float64)

        if not self.cam_init:
            # First frame: initialise camera state based on current VTK camera
            pos = np.array(cam.GetPosition())
            foc = np.array(cam.GetFocalPoint())

            offset = pos - foc
            dist = np.linalg.norm(offset)
            if dist < 1e-6:
                offset = np.array([0.0, 0.0, 5.0])
                dist = np.linalg.norm(offset)

            self.cam_dir = offset / dist
            self.cam_dist = dist
            self.smoothed_center = centre_now.copy()
            self.cam_init = True
        else:
            # Exponential moving average of the cloud centre
            self.smoothed_center = (
                (1.0 - alpha_center) * self.smoothed_center
                + alpha_center * centre_now
            )

        # tiny lateral roll around "up"
        up = np.array([0.0, 1.0, 0.0])
        angle = roll_strength * np.sin(self.phase * 0.3)
        k = up / (np.linalg.norm(up) + 1e-8)
        v = self.cam_dir
        v_cos = v * np.cos(angle)
        v_sin = np.cross(k, v) * np.sin(angle)
        v_par = k * np.dot(k, v) * (1 - np.cos(angle))
        cam_dir_rolled = v_cos + v_sin + v_par

        # New focal point = smoothed centre
        cx, cy, cz = self.smoothed_center
        cam.SetFocalPoint(float(cx), float(cy), float(cz))

        # New position = along rolled direction at fixed distance
        new_pos = self.smoothed_center + cam_dir_rolled * self.cam_dist
        cam.SetPosition(float(new_pos[0]), float(new_pos[1]), float(new_pos[2]))

        # Stable-ish up
        cam.SetViewUp(0.0, 1.0, 0.0)

    # ---- shimmer only ----
    def shimmer(self):
        alpha = 0.72 + 0.18 * np.sin(self.phase)
        actors = self.renderer.GetActors()
        actors.InitTraversal()
        for _ in range(actors.GetNumberOfItems()):
            actor = actors.GetNextActor()
            if actor is not None:
                actor.GetProperty().SetOpacity(alpha)

    # ---- main callback (base version, overridden below) ----
    def Execute(self, obj, event):
        pos_j = self.step_positions()
        self.update_points_and_colors(pos_j)

        centre_now = pos_j.mean(axis=0)
        self.track_camera_inertial(centre_now)

        self.shimmer()
        self.renderer.GetRenderWindow().Render()


# ---------------------------------------------------------------------
# Extended callback: points + blob + links + STARDUST trails
# ---------------------------------------------------------------------
class FullObservatoryCallback(CosmicCloudMorphCallback):

    def __init__(self, points, scalars, poly, clouds3, renderer,
                 blob_actor, link_actor, trail_poly, K):
        super().__init__(points, scalars, poly, clouds3, renderer)
        self.blob_actor = blob_actor
        self.link_actor = link_actor

        # stardust trail geometry
        self.trail_poly = trail_poly
        self.trail_points = trail_poly.GetPoints()
        self.trail_lines = trail_poly.GetLines()

        # one short history per point
        self.trail_histories = [[] for _ in range(K)]
        self.trail_max_len = 20      # steps per particle (tune this)

    def update_stardust_trails(self, pos_j):
        """
        Maintain a short polyline trail for EACH point.
        pos_j: [K, 3] current positions
        """
        K = self.K

        # 1) update histories
        for i in range(K):
            p = np.asarray(pos_j[i], dtype=np.float32)
            self.trail_histories[i].append(p)
            if len(self.trail_histories[i]) > self.trail_max_len:
                self.trail_histories[i] = self.trail_histories[i][-self.trail_max_len:]

        # 2) rebuild vtk geometry
        self.trail_points.Reset()
        self.trail_lines.Reset()

        point_index_offset = 0
        for hist in self.trail_histories:
            n = len(hist)
            if n < 2:
                continue

            # insert this particle's history points
            for h in hist:
                self.trail_points.InsertNextPoint(float(h[0]), float(h[1]), float(h[2]))

            # one polyline through them
            polyline = vtk.vtkPolyLine()
            polyline.GetPointIds().SetNumberOfIds(n)
            for j in range(n):
                polyline.GetPointIds().SetId(j, point_index_offset + j)

            self.trail_lines.InsertNextCell(polyline)
            point_index_offset += n

        self.trail_points.Modified()
        self.trail_lines.Modified()
        self.trail_poly.Modified()

    def Execute(self, obj, event):
        # latent-space morph + jitter
        pos_j = self.step_positions()
        self.update_points_and_colors(pos_j)

        # current positions as numpy (for blob + KNN)
        K = self.K
        pts_now = np.zeros((K, 3), dtype=np.float32)
        for i in range(K):
            pts_now[i] = self.points.GetPoint(i)

        # camera tracks the live cloud with inertia
        centre_now = pts_now.mean(axis=0)
        self.track_camera_inertial(centre_now)

        # update stardust trails for every point
        self.update_stardust_trails(pts_now)

        # update blob (belief field)
        new_blob = make_belief_blob(pts_now)
        self.renderer.RemoveActor(self.blob_actor)
        self.renderer.AddActor(new_blob)
        self.blob_actor = new_blob

        # update KNN links
        new_links = build_knn_edges(pts_now)
        self.renderer.RemoveActor(self.link_actor)
        self.renderer.AddActor(new_links)
        self.link_actor = new_links

        # shimmer everything
        self.shimmer()
        self.renderer.GetRenderWindow().Render()


# ---------------------------------------------------------------------
# Build the full observatory
# ---------------------------------------------------------------------
def build_observatory(clouds3):

    T, K, _ = clouds3.shape
    print(f"[VTK] building belief observatory for T={T}, K={K}")

    pts0 = clouds3[0]

    # --- main point cloud ---
    points = vtk.vtkPoints()
    verts = vtk.vtkCellArray()
    scalars = vtk.vtkIntArray()
    scalars.SetName("quartile")

    centre0 = pts0.mean(axis=0, keepdims=True)
    d0 = np.linalg.norm(pts0 - centre0, axis=1)
    q1, q2, q3 = np.quantile(d0, [0.25, 0.5, 0.75])
    quart0 = np.searchsorted([q1, q2, q3], d0)

    for i in range(K):
        x, y, z = pts0[i]
        pid = points.InsertNextPoint(float(x), float(y), float(z))
        verts.InsertNextCell(1)
        verts.InsertCellPoint(pid)
        scalars.InsertNextValue(int(quart0[i]))

    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetVerts(verts)
    poly.GetPointData().SetScalars(scalars)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    mapper.SetScalarRange(0, 3)
    mapper.SetLookupTable(build_cosmic_lut())
    mapper.SetScalarModeToUsePointData()

    point_actor = vtk.vtkActor()
    point_actor.SetMapper(mapper)
    point_actor.GetProperty().SetPointSize(3.0)

    # --- initial blob + links for t=0 ---
    blob_actor = make_belief_blob(pts0)
    link_actor = build_knn_edges(pts0)

    # --- STARDUST trails: empty polydata, to be populated in callback ---
    trail_points = vtk.vtkPoints()
    trail_lines = vtk.vtkCellArray()

    trail_poly = vtk.vtkPolyData()
    trail_poly.SetPoints(trail_points)
    trail_poly.SetLines(trail_lines)

    trail_mapper = vtk.vtkPolyDataMapper()
    trail_mapper.SetInputData(trail_poly)

    trail_actor = vtk.vtkActor()
    trail_actor.SetMapper(trail_mapper)
    trail_actor.GetProperty().SetColor(1.0, 0.9, 0.7)   # warm dust colour
    trail_actor.GetProperty().SetOpacity(0.55)
    trail_actor.GetProperty().SetLineWidth(1.4)

    # --- renderer setup ---
    ren = vtk.vtkRenderer()
    ren.AddActor(point_actor)
    ren.AddActor(blob_actor)
    ren.AddActor(link_actor)
    ren.AddActor(trail_actor)
    ren.SetBackground(0.01, 0.01, 0.03)

    starfield_actor = build_starfield_actor(num_stars=1000, radius=15.0)
    ren.AddActor(starfield_actor)

    win = vtk.vtkRenderWindow()
    win.AddRenderer(ren)
    win.SetSize(1600, 1000)
    win.SetWindowName("Bayz Belief Observatory")

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(win)

    ren.ResetCamera()
    cam = ren.GetActiveCamera()
    cam.Azimuth(30)
    cam.Elevation(25)
    ren.ResetCameraClippingRange()

    cb = FullObservatoryCallback(points, scalars, poly,
                                 clouds3, ren,
                                 blob_actor, link_actor,
                                 trail_poly, K)

    iren.AddObserver("TimerEvent", cb.Execute)
    iren.CreateRepeatingTimer(TIMER_MS)

    return win, iren



# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    if not os.path.isdir(SNAP_DIR):
        raise FileNotFoundError(f"Snapshot directory not found: {SNAP_DIR}")

    paths = find_snapshot_paths(SNAP_PATTERN)
    clouds = build_all_clouds(paths, NUM_SAMPLES_PER_T)   # [T, K, D]
    clouds3, pca = pca_clouds(clouds)                     # [T, K, 3]

    win, iren = build_observatory(clouds3)
    win.Render()
    iren.Initialize()
    iren.Start()


if __name__ == "__main__":
    main()
