import trimesh
import os
import torch
from pymol import cmd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN
from Bio.PDB import PDBParser, Selection
from data.constants import *
from data.computeHydro import computeHydrophobicity
from data.computeCharges import computeChargeHelper, computeSatisfied_CO_HN
import numpy as np
from sklearn.neighbors import KDTree
import pyvista as pv
from scipy.spatial import Delaunay

def order_point_clouds_globally(fixed_cloud, moving_cloud):
    """
    Find globally optimal point matching to minimize total distance

    Parameters:
    fixed_cloud: torch.Tensor of shape (n, d)
    moving_cloud: torch.Tensor of shape (n, d)

    Returns:
    Reordered moving cloud with minimum total distance
    """
    # Compute pairwise distances
    distances = torch.cdist(fixed_cloud, moving_cloud, p=2.0).cpu().numpy()

    # Use Hungarian algorithm to find optimal matching
    row_ind, col_ind = linear_sum_assignment(distances)

    # Reorder moving cloud based on optimal matching
    reordered_moving = moving_cloud[col_ind]

    return reordered_moving, col_ind

def find_prominent_points(point_cloud, eps=0.5, min_samples=5):

    # Convert to numpy for DBSCAN
    point_cloud_np = point_cloud.cpu().numpy()

    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(point_cloud_np)
    labels = torch.tensor(clustering.labels_, dtype=torch.long, device=point_cloud.device)

    # Number of clusters (excluding noise points with label -1)
    n_clusters = len(set(labels.cpu().numpy())) - (1 if -1 in labels else 0)

    prominent_points = []

    for cluster_label in range(n_clusters):
        # Get points belonging to this cluster
        cluster_mask = labels == cluster_label
        cluster_points = point_cloud[cluster_mask]

        # Compute the convex hull of the cluster (ConvexHull requires numpy array)
        # hull = ConvexHull(cluster_points.cpu().numpy())

        # Find the point with maximum distance from the centroid
        centroid = cluster_points.mean(dim=0)
        distances = torch.norm(cluster_points - centroid, dim=1)
        max_distance_idx = torch.argmax(distances)

        prominent_point = cluster_points[max_distance_idx]
        prominent_points.append(prominent_point)

    prominent_points_tensor = torch.stack(prominent_points)

    return prominent_points_tensor, labels

def get_symmetric_interface_masks(pts_pep, pts_rec, cutoff, max_iterations=1000):
    """
    Get symmetric interface masks where points in both clouds have matching numbers
    of close neighbors, maintaining the original peptide interface size.

    Args:
        pts_pep: (N, 3) tensor of peptide point coordinates
        pts_rec: (M, 3) tensor of receptor point coordinates
        cutoff: Distance cutoff for considering points as neighbors
        max_iterations: Maximum number of iterations to prevent infinite loops

    Returns:
        intf_pep_mask: (N,) boolean tensor marking peptide interface points
        intf_rec_mask: (M,) boolean tensor marking receptor interface points
    """
    # Calculate pairwise distances
    distances = torch.cdist(pts_pep, pts_rec)

    # Get original peptide interface mask
    original_pep_mask = (distances < cutoff).sum(dim=1) > 0
    target_count = torch.sum(original_pep_mask)

    # Create contact matrix
    contact_matrix = (distances < cutoff)  # (N, M) boolean matrix

    # Get number of contacts for each point
    pep_contacts = contact_matrix.sum(dim=1)  # (N,)
    rec_contacts = contact_matrix.sum(dim=0)  # (M,)

    # Initial masks for points with any contacts
    intf_pep_mask = (pep_contacts > 0)
    intf_rec_mask = (rec_contacts > 0)

    # Check if target count is achievable
    max_possible_contacts = torch.sum(pep_contacts > 0)
    if target_count > max_possible_contacts:
        target_count = max_possible_contacts
        print('torch.sum(original_pep_mask) > max_possible_contacts')

    # Adjust receptor mask
    iteration_count = 0
    prev_rec_count = -1

    while torch.sum(intf_rec_mask) != target_count:
        current_rec_count = torch.sum(intf_rec_mask)

        # Check for stuck condition
        if current_rec_count == prev_rec_count:
            break
        prev_rec_count = current_rec_count

        # Check iteration limit
        iteration_count += 1
        if iteration_count > max_iterations:
            break

        if torch.sum(intf_rec_mask) > target_count:
            # Remove receptor point with fewest contacts
            rec_contact_counts = rec_contacts.clone()
            rec_contact_counts[~intf_rec_mask] = 9999999
            idx_to_remove = torch.argmin(rec_contact_counts)
            intf_rec_mask[idx_to_remove] = False
        else:
            # If we have too few points, we need to add back points
            rec_contact_counts = rec_contacts.clone()
            rec_contact_counts[intf_rec_mask] = 0
            if torch.max(rec_contact_counts) == 0:  # No more points can be added
                break
            idx_to_add = torch.argmax(rec_contact_counts)
            intf_rec_mask[idx_to_add] = True

    # Adjust peptide mask
    iteration_count = 0
    prev_pep_count = -1

    while torch.sum(intf_pep_mask) != target_count:
        current_pep_count = torch.sum(intf_pep_mask)

        # Check for stuck condition
        if current_pep_count == prev_pep_count:
            break
        prev_pep_count = current_pep_count

        # Check iteration limit
        iteration_count += 1
        if iteration_count > max_iterations:
            break

        if torch.sum(intf_pep_mask) > target_count:
            # Remove peptide point with fewest contacts
            pep_contact_counts = pep_contacts.clone()
            pep_contact_counts[~intf_pep_mask] = 9999999
            idx_to_remove = torch.argmin(pep_contact_counts)
            intf_pep_mask[idx_to_remove] = False
        else:
            # If we have too few points, we need to add back points
            pep_contact_counts = pep_contacts.clone()
            pep_contact_counts[intf_pep_mask] = 0
            if torch.max(pep_contact_counts) == 0:  # No more points can be added
                break
            idx_to_add = torch.argmax(pep_contact_counts)
            intf_pep_mask[idx_to_add] = True

    # Ensure equal counts in final masks
    final_count = min(torch.sum(intf_pep_mask), torch.sum(intf_rec_mask))
    if torch.sum(intf_pep_mask) > final_count:
        pep_contact_counts = pep_contacts.clone()
        pep_contact_counts[~intf_pep_mask] = 9999999
        while torch.sum(intf_pep_mask) > final_count:
            idx_to_remove = torch.argmin(pep_contact_counts)
            intf_pep_mask[idx_to_remove] = False
            pep_contact_counts[idx_to_remove] = 9999999

    if torch.sum(intf_rec_mask) > final_count:
        rec_contact_counts = rec_contacts.clone()
        rec_contact_counts[~intf_rec_mask] = 9999999
        while torch.sum(intf_rec_mask) > final_count:
            idx_to_remove = torch.argmin(rec_contact_counts)
            intf_rec_mask[idx_to_remove] = False
            rec_contact_counts[idx_to_remove] = 9999999

    return intf_pep_mask, intf_rec_mask

def load_point_cloud_by_file_extension(file_name, with_normal=True):
    mesh = trimesh.load(file_name, force='mesh')
    point_set = torch.tensor(mesh.vertices).float()

    if with_normal:
        vertices_normal = torch.tensor(mesh.vertex_normals).float()
        return point_set, vertices_normal
    return point_set


def subsample_points(points, proportion=0.5, method='random'):
    """
    Subsample points by proportion of original data

    Args:
        points: (N, 3) tensor of points
        proportion: float between 0 and 1, proportion of points to keep
        method: 'random' or 'fps' (farthest point sampling)

    Returns:
        (M, 3) tensor of subsampled points, where M = N * proportion
    """
    if not 0 < proportion <= 1:
        raise ValueError("Proportion must be between 0 and 1")

    target_size = int(len(points) * proportion)

    if method == 'random':
        idx = torch.randperm(len(points))[:target_size]
        return points[idx]

    elif method == 'fps':
        # Farthest Point Sampling
        fps_idx = torch.zeros(target_size, dtype=torch.long)
        distances = torch.ones(len(points)) * 1e10

        # Pick first point randomly
        fps_idx[0] = torch.randint(len(points), (1,))

        for i in range(1, target_size):
            # Find distances to last chosen point
            last_point = points[fps_idx[i - 1]]
            dist = torch.sum((points - last_point) ** 2, dim=1)
            distances = torch.min(distances, dist)

            # Pick point with maximum distance
            fps_idx[i] = torch.max(distances, dim=0)[1]

        return points[fps_idx]


def gaussian_kernel_smoothing(points, sigma=1.0, k_neighbors=16):
    """
    Apply Gaussian kernel smoothing to point cloud

    Args:
        points: (N, 3) tensor of points
        sigma: bandwidth parameter for Gaussian kernel
        k_neighbors: number of neighbors for local smoothing

    Returns:
        (N, 3) tensor of smoothed points
    """
    # Convert to numpy for KDTree
    points_np = points.numpy()

    # Build KD-tree
    tree = KDTree(points_np)

    # Find k nearest neighbors for each point
    distances, indices = tree.query(points_np, k=k_neighbors)

    # Compute Gaussian weights
    weights = np.exp(-distances ** 2 / (2 * sigma ** 2))
    weights = weights / weights.sum(axis=1, keepdims=True)

    # Apply weighted average
    smoothed_points = torch.zeros_like(points)
    for i in range(len(points)):
        neighbor_points = points[indices[i]]
        smoothed_points[i] = torch.sum(neighbor_points * torch.tensor(weights[i]).unsqueeze(1), dim=0)

    return smoothed_points


def preprocess_surface_single(structure_dir, pdb_fname, type):

    # try:
    # pdb_path = os.path.join(structure_dir, pdb_fname)
    pdb_path = structure_dir
    pep_path = os.path.join(pdb_path, type)
    # pep = parse_pdb(pdb_path)[0]

    # surf_path1 = os.path.join(pdb_path, f'surface_{pdb_fname}_.obj')
    # surf_path2 = os.path.join(pdb_path, f'surface_poc_{pdb_fname}_.obj')
    # if os.path.isfile(surf_path1):
    #     os.remove(surf_path1)
    # if os.path.isfile(surf_path2):
    #     os.remove(surf_path2)
    surf_path = os.path.join(pdb_path, f'surface_{pdb_fname}_{type}.obj')
    if not os.path.exists(surf_path) or os.path.getsize(surf_path) == 0:
        cmd.reinitialize()
        cmd.load(pep_path)
        cmd.show_as('surface')
        cmd.set_view((1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 300, 1))  # adjust the coordinates of vertices
        cmd.save(surf_path)
    pts, norms = load_point_cloud_by_file_extension(surf_path)

    # Smooth
    # sigma = 1.0
    # k_neighbors = 16
    # smoothed_points = gaussian_kernel_smoothing(pts, sigma, k_neighbors)
    #
    # # Subsample
    # subsampled_points = subsample_points(smoothed_points, proportion = 0.5, method='fps')
    subsampled_points = pts

    hp, hbond = computer_property(pep_path, subsampled_points)

    return subsampled_points, norms, hp, hbond

def plot_surface(xyz, save_path, type, distance_threshold = 5):

    points = xyz.detach().cpu().numpy() if torch.is_tensor(xyz) else xyz

    tri = Delaunay(points)

    # Get edges from triangulation
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                edge = tuple(sorted([simplex[i], simplex[j]]))
                edges.add(edge)

    # Filter edges based on distance
    filtered_edges = []
    filtered_simplices = []

    for simplex in tri.simplices:
        valid_simplex = True
        for i in range(3):
            for j in range(i + 1, 3):
                p1 = points[simplex[i]]
                p2 = points[simplex[j]]
                distance = np.linalg.norm(p1 - p2)
                if distance > distance_threshold:
                    valid_simplex = False
                    break
        if valid_simplex:
            filtered_simplices.append(simplex)

    if not filtered_simplices:
        raise ValueError("No simplices remain after distance filtering")

    filtered_simplices = np.array(filtered_simplices)

    # Create faces array for PyVista
    faces = np.column_stack((np.full(len(filtered_simplices), 4), filtered_simplices)).ravel()
    # faces = np.column_stack((np.full(tri.simplices.shape[0], 4), tri.simplices)).ravel()

    mesh = pv.PolyData(points, faces)

    center = mesh.center
    center_point = pv.PolyData(center)
    center_point.point_data['labels'] = [type]
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='lightblue', show_edges=True)
    plotter.add_point_labels(center_point, 'labels', point_size=20, font_size=20)

    mesh.save(f'{save_path}/mesh_{type}.vtk')


    # mesh.plot(point_size=10, style='points', show_edges=True)

    # mesh.save(f'{save_path}/mesh_{type}.vtk', dpi=600)
    # plotter = pv.Plotter()
    # plotter.add_mesh(mesh, show_edges=True)
    # plotter.screenshot("delaunay_mesh.png")  # Save the image as a .png file


def computer_property(path, pts):
    Tensor, tensor = torch.LongTensor, torch.FloatTensor
    parser = PDBParser()
    structure = parser.get_structure(None, path)
    residues, atom_xyz, atom_types, res_types = [], [], [], []
    for i, res in enumerate(structure.get_residues()):
        resname = res.get_resname()
        if not AA.is_aa(resname) or not (res.has_id('CA') and res.has_id('C') and res.has_id('N')): continue

        resname = non_standard_residue_substitutions.get(resname, resname)
        for atom in res:
            atom_name = atom.get_name()
            if atom_name != '':
                atom_xyz.append(atom.get_coord())
                atom_types.append(atom)
                residues.append(res)
                res_types.append(resname)
    atom_xyz, res_types, atom_types = tensor(np.stack(atom_xyz)), np.array(res_types), np.array(atom_types)

    # compute hydrophobicity
    # knn_res_idx = torch.cdist(pts.cuda(), atom_xyz.cuda()).topk(1, dim=1, largest=False)[1].squeeze().cpu()   # accelerate on GPU
    knn_res_idx = torch.cdist(pts, atom_xyz).topk(1, dim=1, largest=False)[1].squeeze()
    knn_res_types = res_types[knn_res_idx].tolist()
    hp: Tensor = computeHydrophobicity(knn_res_types)

    # compute charge
    hbond = torch.zeros(len(pts))
    atoms = Selection.unfold_entities(structure, "A")   # unfold into atom level (A)
    satisfied_CO, satisfied_HN = computeSatisfied_CO_HN(atoms)
    knn_atom_idx = torch.cdist(pts, atom_xyz).topk(1, dim=1, largest=False)[1].squeeze()
    for ix in range(len(pts)):
        res = residues[knn_res_idx[ix]]
        res_id = res.get_id()
        atom = atom_types[knn_atom_idx[ix]]
        res_name = knn_res_types[ix]
        if not (atom.element == 'H' and res_id in satisfied_HN) and not (atom.element == 'O' and res_id in satisfied_CO):
            hbond[ix] = computeChargeHelper(atom.get_name(), res, res_name, pts[ix])  # Ignore atom if it is BB

        # if not (try_get_element(atom) == 'H' and res_id in satisfied_HN) and not (try_get_element(atom) == 'O' and res_id in satisfied_CO):
        #     hbond[ix] = computeChargeHelper(atom.get_name(), res, res_name, pts[ix])

    return hp, hbond