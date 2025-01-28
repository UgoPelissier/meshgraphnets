import shutil
import os
from tqdm import tqdm
import numpy as np
import pyvista as pv
import meshio
import torch
from torch_geometric.data import Data
from typing import List


def convert_to_meshio_vtu(graph: Data) -> meshio.Mesh:
    """
    Converts a PyTorch Geometric graph to a Meshio mesh.

    Args:
        graph (Data): The graph data to convert.

    Returns:
        meshio.Mesh: The converted Meshio mesh.
    """
    # Extract node positions and ensure they have three coordinates
    vertices = graph.mesh_pos.cpu().numpy()
    num_coords = vertices.shape[1]
    if num_coords < 3:
        # Pad with zeros to make it 3D
        padding = np.zeros((vertices.shape[0], 3 - num_coords), dtype=vertices.dtype)
        vertices = np.hstack([vertices, padding])
    elif num_coords > 3:
        raise ValueError(f"Unsupported vertex dimension: {num_coords}")
    
    # Extract faces
    cells = [("triangle", graph.cells.cpu().numpy())]
    
    # Create Meshio mesh
    mesh = meshio.Mesh(vertices, cells)

    # Optionally add all node features as point data
    mesh.point_data['Vitesse'] = graph.x[:,:2].cpu().numpy()
    # mesh.point_data['Node Type'] = torch.argmax(graph.x[:,5:],dim=1).cpu().numpy()
    
    return mesh

def vtu_to_xdmf(filename: str, filelist: List[str], timestep=1) -> None:
    """
    Writes a time series of meshes (same points and cells) into XDMF/HDF5 format.
    
    Args:
        filename (str): Name for the XDMF/HDF5 file without the extension
        filelist(List[str]): List of the files' paths to compress.

    Returns:
        None: XDMF/HDF5 file is saved to the path filename.
    """
    h5_filename = f"{filename}.h5"
    xdmf_filename = f"{filename}.xdmf"

    init_vtu = meshio.read(filelist[0])
    points = init_vtu.points
    cells = init_vtu.cells

    # Open the TimeSeriesWriter for HDF5
    with meshio.xdmf.TimeSeriesWriter(xdmf_filename) as writer:
        # Write the mesh (points and cells) once
        writer.write_points_cells(points, cells)

        # Loop through time steps and write data
        t = 0
        for file in tqdm(filelist, desc='Compressing VTUs into XDMF files'):
            mesh = meshio.read(file)
            point_data = mesh.point_data
            cell_data = mesh.cell_data
            writer.write_data(t, point_data=point_data, cell_data=cell_data)
            t += timestep

    # The H5 archive is systematically created in cwd, we just need to move it
    shutil.move(src=os.path.join(os.getcwd(), os.path.split(h5_filename)[1]), dst=h5_filename)
    print(f"Time series written to {xdmf_filename} and {h5_filename}")

    # Remove the temporary files
    for file in filelist:
        os.remove(file)