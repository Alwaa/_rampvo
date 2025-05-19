import numpy as np
import g2o

def bundle_adjustment(R_dict: dict,
                      t_dict: dict,
                      max_iterations: int = 15,
                      weights = None
                      ) -> dict:
    """
    Pose-graph bundle adjustment over a set of relative pose constraints.

    Args:
        R_dict: dict mapping (i,j) -> 3x3 relative rotation from i to j
        t_dict: dict mapping (i,j) -> 3-vector relative translation (unit)
        max_iterations: number of optimizer iterations

    Returns:
        abs_poses: dict mapping idx -> (R, t)
                   where R is 3x3 rotation, t is 3-vector translation
    """
    # Collect all unique vertex indices
    idxs = sorted({i for (i, j) in R_dict.keys()} | {j for (i, j) in R_dict.keys()})

    # Initialize absolute pose estimates
    abs_estimates = {}
    # seed first vertex at origin: as Isometry3d
    R0 = np.eye(3)
    t0 = np.zeros((3,))
    abs_estimates[idxs[0]] = g2o.Isometry3d(R0, t0)

    # propagate initial chain
    for (i, j), R in R_dict.items():
        if i in abs_estimates:
            # get previous absolute pose
            iso_i = abs_estimates[i]
            # relative translation
            t_raw = t_dict[(i, j)]
            t_vec = np.array(t_raw).flatten()[:3]
            # compose transforms: Iso_j = Iso_i * Rel
            rel_iso = g2o.Isometry3d(R, t_vec)
            abs_estimates[j] = iso_i * rel_iso

    # Setup optimizer
    optimizer = g2o.SparseOptimizer()
    linear_solver = g2o.LinearSolverDenseSE3()
    block_solver = g2o.BlockSolverSE3(linear_solver)
    algorithm = g2o.OptimizationAlgorithmLevenberg(block_solver)
    optimizer.set_algorithm(algorithm)

    # Add vertices
    for idx in idxs:
        v = g2o.VertexSE3()
        v.set_id(idx)
        est_iso = abs_estimates.get(idx, g2o.Isometry3d(np.eye(3), np.zeros((3,))))
        v.set_estimate(est_iso)
        if idx == idxs[0]:
            v.set_fixed(True)
        optimizer.add_vertex(v)

    # Add edges
    for (i, j), R in R_dict.items():
        t_vec = np.array(t_dict[(i, j)]).flatten()[:3]
        edge = g2o.EdgeSE3()
        edge.set_vertex(0, optimizer.vertex(i))
        edge.set_vertex(1, optimizer.vertex(j))
        meas = g2o.Isometry3d(R, t_vec)
        edge.set_measurement(meas)

        if weights is None:
            edge.set_information(np.eye(6))
        else:
            # weights[(i,j)] is an (Ni_j,2) array but you want a single scalar per edge:
            w_ij = float(weights[(i,j)].mean())  # average confidence over that edge
            # build a diagonal 6×6 info matrix:
            info = np.diag([w_ij, w_ij, w_ij, 1e-3, 1e-3, 1e-3])
            edge.set_information(info)

        optimizer.add_edge(edge)

    # Optimize
    optimizer.initialize_optimization()
    optimizer.optimize(max_iterations)

    # Extract optimized poses
    abs_poses = {}
    for idx in idxs:
        v = optimizer.vertex(idx)
        iso = v.estimate()  # Isometry3d
        R_opt = iso.rotation().matrix()
        t_opt = iso.translation()
        abs_poses[idx] = (R_opt, t_opt)

    return abs_poses




def __bundle_adjustment(R_dict: dict,
                      t_dict: dict,
                      max_iterations: int = 15
                      ) -> dict:
    """
    Pose-graph bundle adjustment over a set of relative pose constraints.

    Args:
        R_dict: dict mapping (i,j) -> 3x3 relative rotation from i to j
        t_dict: dict mapping (i,j) -> 3-vector relative translation (unit)
        max_iterations: number of optimizer iterations

    Returns:
        abs_poses: dict mapping idx -> (R, t)
                   where R is 3x3 rotation, t is 3-vector translation
    """
    # Collect all unique vertex indices
    idxs = set()
    for (i, j) in R_dict.keys():
        idxs.add(i)
        idxs.add(j)
    idxs = sorted(idxs)

    # Initialize absolute poses by chaining in index order
    abs_estimates = {}
    # seed first vertex at origin (rotation identity, translation zero column)
    abs_estimates[idxs[0]] = g2o.SE3Quat(np.eye(3), np.zeros((3,1)))
    # propagate initial values
    for (i, j), R in R_dict.items():
        if i in abs_estimates:
            t = t_dict[(i, j)].reshape(3, 1)
            abs_estimates[j] = abs_estimates[i] * g2o.SE3Quat(R, t)

    # Setup optimizer
    optimizer = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
    algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer.set_algorithm(algorithm)

    # Add SE3 vertices
    for idx in idxs:
        v = g2o.VertexSE3()
        v.set_id(idx)
        # ensure each SE3Quat estimate has t as (3,1)
        est = abs_estimates.get(idx, g2o.SE3Quat(np.eye(3), np.zeros((3,1))))
        v.set_estimate(est)
        if idx == idxs[0]:
            v.set_fixed(True)
        optimizer.add_vertex(v)

    # Add relative-pose edges
    for (i, j), R in R_dict.items():
        t = t_dict[(i, j)].reshape(3, 1)
        edge = g2o.EdgeSE3()
        edge.set_vertex(0, optimizer.vertex(i))
        edge.set_vertex(1, optimizer.vertex(j))
        measurement = g2o.SE3Quat(R, t)
        edge.set_measurement(measurement)
        edge.set_information(np.eye(6))
        optimizer.add_edge(edge)

    # Optimize
    optimizer.initialize_optimization()
    optimizer.optimize(max_iterations)

    # Extract optimized poses
    abs_poses = {}
    for idx in idxs:
        v = optimizer.vertex(idx)
        est = v.estimate()
        # convert to rotation matrix and translation vector
        R_opt = est.rotation().matrix()
        t_opt = est.translation().reshape(3,)  # flatten (3,1) to (3,)
        abs_poses[idx] = (R_opt, t_opt)

    return abs_poses


def _old_bundle_adjustment(R_dict: dict,
                      t_dict: dict,
                      max_iterations: int = 15
                      ) -> dict:
    """
    Pose-graph bundle adjustment over a set of relative pose constraints.

    Args:
        R_dict: dict mapping (i,j) -> 3x3 relative rotation from i to j
        t_dict: dict mapping (i,j) -> 3-vector relative translation (unit)
        max_iterations: number of optimizer iterations

    Returns:
        abs_poses: dict mapping idx -> (R, t)
                   where R is 3x3 rotation, t is 3-vector translation
    """
    # Collect all unique vertex indices
    idxs = set()
    for (i, j) in R_dict.keys():
        idxs.add(i)
        idxs.add(j)
    idxs = sorted(idxs)

    # Initialize absolute poses by chaining in index order
    abs_estimates = {}
    # seed first vertex at origin
    abs_estimates[idxs[0]] = g2o.SE3Quat(np.eye(3), np.zeros(3))
    # propagate initial values
    for (i, j), R in R_dict.items():
        if i in abs_estimates:
            t = t_dict[(i, j)]
            abs_estimates[j] = abs_estimates[i] * g2o.SE3Quat(R, t)

    # Setup optimizer
    optimizer = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
    algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer.set_algorithm(algorithm)

    # Add SE3 vertices
    for idx in idxs:
        v = g2o.VertexSE3()
        v.set_id(idx)
        v.set_estimate(abs_estimates.get(idx, g2o.SE3Quat()))
        if idx == idxs[0]:
            v.set_fixed(True)
        optimizer.add_vertex(v)

    # Add relative-pose edges
    for (i, j), R in R_dict.items():
        t = t_dict[(i, j)]
        edge = g2o.EdgeSE3()
        edge.set_vertex(0, optimizer.vertex(i))
        edge.set_vertex(1, optimizer.vertex(j))
        measurement = g2o.SE3Quat(R, t)
        edge.set_measurement(measurement)
        edge.set_information(np.eye(6))
        optimizer.add_edge(edge)

    # Optimize
    optimizer.initialize_optimization()
    optimizer.optimize(max_iterations)

    # Extract optimized poses
    abs_poses = {}
    for idx in idxs:
        v = optimizer.vertex(idx)
        est = v.estimate()
        # convert to rotation matrix and translation
        R_opt = est.rotation().matrix()
        t_opt = est.translation()
        abs_poses[idx] = (R_opt, t_opt)

    return abs_poses
