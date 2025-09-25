import vtk
import numpy as np
import random


def sample_random_unit_vector():
    vec = np.random.normal(size=3)
    return vec / np.linalg.norm(vec)


def get_cell_centroid(mesh, cell_id):
    cell = mesh.GetCell(cell_id)
    pts = cell.GetPoints()
    return np.mean([np.array(pts.GetPoint(i)) for i in range(pts.GetNumberOfPoints())], axis=0)


def get_cell_normal(mesh, cell_id):
    cell = mesh.GetCell(cell_id)
    pts = cell.GetPoints()
    p0 = np.array(pts.GetPoint(0))
    p1 = np.array(pts.GetPoint(1))
    p2 = np.array(pts.GetPoint(2))
    normal = np.cross(p1 - p0, p2 - p0)
    return normal / (np.linalg.norm(normal) + 1e-8)


def compute_cell_area(mesh, cell_id):
    cell = mesh.GetCell(cell_id)
    pts = cell.GetPoints()
    p0, p1, p2 = [np.array(pts.GetPoint(i)) for i in range(3)]
    return 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))


def extract_partial_surface_stl(input_stl_path, output_stl_path,
                                surface_amount_min=0.1, surface_amount_max=0.5,
                                w_distance=1.0, w_normal=0.3, w_noise=0.2,
                                seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    reader = vtk.vtkSTLReader()
    reader.SetFileName(input_stl_path)
    reader.Update()
    mesh = reader.GetOutput()

    num_cells = mesh.GetNumberOfCells()
    if num_cells == 0:
        raise RuntimeError("Empty mesh!")

    center = np.mean([np.array(mesh.GetPoint(i)) for i in range(mesh.GetNumberOfPoints())], axis=0)
    random_dir = sample_random_unit_vector()
    surface_amount = random.uniform(surface_amount_min, surface_amount_max)

    total_area = sum(compute_cell_area(mesh, cid) for cid in range(num_cells))
    target_area = surface_amount * total_area
    print(f"🎯 Target surface area = {surface_amount:.3f} × {total_area:.2f} = {target_area:.2f}")
    print(f"🧭 Random extraction direction: {random_dir}")

    # 建立邻接结构
    adjacency = {cid: set() for cid in range(num_cells)}
    for pid in range(mesh.GetNumberOfPoints()):
        cell_ids = vtk.vtkIdList()
        mesh.GetPointCells(pid, cell_ids)
        ids = [cell_ids.GetId(i) for i in range(cell_ids.GetNumberOfIds())]
        for i in ids:
            for j in ids:
                if i != j:
                    adjacency[i].add(j)

    visited = set()
    selected = set()
    area_accum = 0.0

    seed_cell = random.randint(0, num_cells - 1)
    queue = [seed_cell]
    visited.add(seed_cell)

    while queue and area_accum < target_area:
        cid = queue.pop(0)
        selected.add(cid)
        area_accum += compute_cell_area(mesh, cid)

        for nid in adjacency[cid]:
            if nid not in visited:
                visited.add(nid)

                centroid = get_cell_centroid(mesh, nid)
                normal = get_cell_normal(mesh, nid)
                dist_score = np.linalg.norm(centroid - center)
                normal_score = np.dot(normal, random_dir)
                noise_score = random.random()
                score = w_distance * dist_score - w_normal * normal_score + w_noise * noise_score

                if score < 1.5:  # 控制生长的阈值，越小越“局部”
                    queue.append(nid)

    print(f"✅ Selected {len(selected)} triangles, total area ≈ {area_accum:.2f}")

    ids = vtk.vtkIdTypeArray()
    for cid in selected:
        ids.InsertNextValue(cid)

    selection_node = vtk.vtkSelectionNode()
    selection_node.SetFieldType(vtk.vtkSelectionNode.CELL)
    selection_node.SetContentType(vtk.vtkSelectionNode.INDICES)
    selection_node.SetSelectionList(ids)

    selection = vtk.vtkSelection()
    selection.AddNode(selection_node)

    extract_selection = vtk.vtkExtractSelection()
    extract_selection.SetInputData(0, mesh)
    extract_selection.SetInputData(1, selection)
    extract_selection.Update()

    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputConnection(extract_selection.GetOutputPort())
    geo_filter.Update()

    partial_mesh = geo_filter.GetOutput()

    writer = vtk.vtkSTLWriter()
    writer.SetFileName(output_stl_path)
    writer.SetInputData(partial_mesh)
    writer.Write()

    print(f"💾 Saved partial mesh to: {output_stl_path}")


# === 示例调用 ===
if __name__ == "__main__":
    extract_partial_surface_stl(
        input_stl_path="surface.stl",
        output_stl_path="partial_output.stl",
        surface_amount_min=0.2,
        surface_amount_max=0.4,
        w_distance=1.0,
        w_normal=0.5,
        w_noise=0.3,
        seed=42
    )
