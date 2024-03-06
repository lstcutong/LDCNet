import torch
import sys
import torch.nn as nn
from pointpd.models.builder import MODELS
import numpy as np
import os
from os.path import join, exists
from os import makedirs
import enum
from pointpd.datasets.helper import write_ply, read_ply

DIR = os.path.dirname(os.path.realpath(__file__))

def fitting_loss(sq_distance, radius):
    """KPConv fitting loss. For each query point it ensures that at least one neighboor is
    close to each kernel point

    Arguments:
        sq_distance - For each querry point, from all neighboors to all KP points [N_querry, N_neighboors, N_KPoints]
        radius - Radius of the convolution
    """
    kpmin = sq_distance.min(dim=1)[0]
    normalised_kpmin = kpmin / (radius**2)
    return torch.mean(normalised_kpmin)


def repulsion_loss(deformed_kpoints, radius):
    """Ensures that the deformed points within the kernel remain equidistant

    Arguments:
        deformed_kpoints - deformed points for each query point
        radius - Radius of the kernel
    """
    deformed_kpoints / float(radius)
    n_points = deformed_kpoints.shape[1]
    repulsive_loss = 0
    for i in range(n_points):
        with torch.no_grad():
            other_points = torch.cat([deformed_kpoints[:, :i, :], deformed_kpoints[:, i + 1 :, :]], dim=1)
        distances = torch.sqrt(torch.sum((other_points - deformed_kpoints[:, i : i + 1, :]) ** 2, dim=-1))
        repulsion_force = torch.sum(torch.pow(torch.relu(1.5 - distances), 2), dim=1)
        repulsive_loss += torch.mean(repulsion_force)
    return repulsive_loss


def permissive_loss(deformed_kpoints, radius):
    """This loss is responsible to penalize deformed_kpoints to
    move outside from the radius defined for the convolution
    """
    norm_deformed_normalized = torch.norm(deformed_kpoints, p=2, dim=-1) / float(radius)
    permissive_loss = torch.mean(norm_deformed_normalized[norm_deformed_normalized > 1.0])
    return permissive_loss

def radius_gaussian(sq_r, sig, eps=1e-9):
    """
    Compute a radius gaussian (gaussian of distance)
    :param sq_r: input radiuses [dn, ..., d1, d0]
    :param sig: extents of gaussians [d1, d0] or [d0] or float
    :return: gaussian of sq_r [dn, ..., d1, d0]
    """
    return torch.exp(-sq_r / (2 * sig**2 + eps))

class ConvolutionFormat(enum.Enum):
    DENSE = "dense"
    PARTIAL_DENSE = "partial_dense"
    MESSAGE_PASSING = "message_passing"
    SPARSE = "sparse"

def kernel_point_optimization_debug(
    radius, num_points, num_kernels=1, dimension=3, fixed="center", ratio=1.0, verbose=0
):
    """
    Creation of kernel point via optimization of potentials.
    :param radius: Radius of the kernels
    :param num_points: points composing kernels
    :param num_kernels: number of wanted kernels
    :param dimension: dimension of the space
    :param fixed: fix position of certain kernel points ('none', 'center' or 'verticals')
    :param ratio: ratio of the radius where you want the kernels points to be placed
    :param verbose: display option
    :return: points [num_kernels, num_points, dimension]
    """

    #######################
    # Parameters definition
    #######################

    # Radius used for optimization (points are rescaled afterwards)
    radius0 = 1
    diameter0 = 2

    # Factor multiplicating gradients for moving points (~learning rate)
    moving_factor = 1e-2
    continuous_moving_decay = 0.9995

    # Gradient threshold to stop optimization
    thresh = 1e-5

    # Gradient clipping value
    clip = 0.05 * radius0

    #######################
    # Kernel initialization
    #######################

    # Random kernel points
    kernel_points = np.random.rand(num_kernels * num_points - 1, dimension) * diameter0 - radius0
    while kernel_points.shape[0] < num_kernels * num_points:
        new_points = np.random.rand(num_kernels * num_points - 1, dimension) * diameter0 - radius0
        kernel_points = np.vstack((kernel_points, new_points))
        d2 = np.sum(np.power(kernel_points, 2), axis=1)
        kernel_points = kernel_points[d2 < 0.5 * radius0 * radius0, :]
    kernel_points = kernel_points[: num_kernels * num_points, :].reshape((num_kernels, num_points, -1))

    # Optionnal fixing
    if fixed == "center":
        kernel_points[:, 0, :] *= 0
    if fixed == "verticals":
        kernel_points[:, :3, :] *= 0
        kernel_points[:, 1, -1] += 2 * radius0 / 3
        kernel_points[:, 2, -1] -= 2 * radius0 / 3

    #####################
    # Kernel optimization
    #####################

    # Initiate figure

    saved_gradient_norms = np.zeros((10000, num_kernels))
    old_gradient_norms = np.zeros((num_kernels, num_points))
    for iter in range(10000):

        # Compute gradients
        # *****************

        # Derivative of the sum of potentials of all points
        A = np.expand_dims(kernel_points, axis=2)
        B = np.expand_dims(kernel_points, axis=1)
        interd2 = np.sum(np.power(A - B, 2), axis=-1)
        inter_grads = (A - B) / (np.power(np.expand_dims(interd2, -1), 3 / 2) + 1e-6)
        inter_grads = np.sum(inter_grads, axis=1)

        # Derivative of the radius potential
        circle_grads = 10 * kernel_points

        # All gradients
        gradients = inter_grads + circle_grads

        if fixed == "verticals":
            gradients[:, 1:3, :-1] = 0

        # Stop condition
        # **************

        # Compute norm of gradients
        gradients_norms = np.sqrt(np.sum(np.power(gradients, 2), axis=-1))
        saved_gradient_norms[iter, :] = np.max(gradients_norms, axis=1)

        # Stop if all moving points are gradients fixed (low gradients diff)

        if fixed == "center" and np.max(np.abs(old_gradient_norms[:, 1:] - gradients_norms[:, 1:])) < thresh:
            break
        elif fixed == "verticals" and np.max(np.abs(old_gradient_norms[:, 3:] - gradients_norms[:, 3:])) < thresh:
            break
        elif np.max(np.abs(old_gradient_norms - gradients_norms)) < thresh:
            break
        old_gradient_norms = gradients_norms

        # Move points
        # ***********

        # Clip gradient to get moving dists
        moving_dists = np.minimum(moving_factor * gradients_norms, clip)

        # Fix central point
        if fixed == "center":
            moving_dists[:, 0] = 0
        if fixed == "verticals":
            moving_dists[:, 0] = 0

        # Move points
        kernel_points -= np.expand_dims(moving_dists, -1) * gradients / np.expand_dims(gradients_norms + 1e-6, -1)



        # moving factor decay
        moving_factor *= continuous_moving_decay

    # Rescale radius to fit the wanted ratio of radius
    r = np.sqrt(np.sum(np.power(kernel_points, 2), axis=-1))
    kernel_points *= ratio / np.mean(r[:, 1:])

    # Rescale kernels with real radius
    return kernel_points * radius, saved_gradient_norms


def load_kernels(radius, num_kpoints, num_kernels, dimension, fixed):

    # Number of tries in the optimization process, to ensure we get the most stable disposition
    num_tries = 100

    # Kernel directory
    kernel_dir = join(DIR, "kernels/dispositions")
    if not exists(kernel_dir):
        makedirs(kernel_dir)

    # Kernel_file
    if dimension == 3:
        kernel_file = join(kernel_dir, "k_{:03d}_{:s}.ply".format(num_kpoints, fixed))
    elif dimension == 2:
        kernel_file = join(kernel_dir, "k_{:03d}_{:s}_2D.ply".format(num_kpoints, fixed))
    else:
        raise ValueError("Unsupported dimpension of kernel : " + str(dimension))

    # Check if already done
    if not exists(kernel_file):

        # Create kernels
        kernel_points, grad_norms = kernel_point_optimization_debug(
            1.0,
            num_kpoints,
            num_kernels=num_tries,
            dimension=dimension,
            fixed=fixed,
            verbose=0,
        )

        # Find best candidate
        best_k = np.argmin(grad_norms[-1, :])

        # Save points
        original_kernel = kernel_points[best_k, :, :]
        write_ply(kernel_file, original_kernel, ["x", "y", "z"])

    else:
        data = read_ply(kernel_file)
        original_kernel = np.vstack((data["x"], data["y"], data["z"])).T

    # N.B. 2D kernels are not supported yet
    if dimension == 2:
        return original_kernel

    # Random rotations depending of the fixed points
    if fixed == "verticals":

        # Create random rotations
        thetas = np.random.rand(num_kernels) * 2 * np.pi
        c, s = np.cos(thetas), np.sin(thetas)
        R = np.zeros((num_kernels, 3, 3), dtype=np.float32)
        R[:, 0, 0] = c
        R[:, 1, 1] = c
        R[:, 2, 2] = 1
        R[:, 0, 1] = s
        R[:, 1, 0] = -s

        # Scale kernels
        original_kernel = radius * np.expand_dims(original_kernel, 0)

        # Rotate kernels
        kernels = np.matmul(original_kernel, R)

    else:

        # Create random rotations
        u = np.ones((num_kernels, 3))
        v = np.ones((num_kernels, 3))
        wrongs = np.abs(np.sum(u * v, axis=1)) > 0.99
        while np.any(wrongs):
            new_u = np.random.rand(num_kernels, 3) * 2 - 1
            new_u = new_u / np.expand_dims(np.linalg.norm(new_u, axis=1) + 1e-9, -1)
            u[wrongs, :] = new_u[wrongs, :]
            new_v = np.random.rand(num_kernels, 3) * 2 - 1
            new_v = new_v / np.expand_dims(np.linalg.norm(new_v, axis=1) + 1e-9, -1)
            v[wrongs, :] = new_v[wrongs, :]
            wrongs = np.abs(np.sum(u * v, axis=1)) > 0.99

        # Make v perpendicular to u
        v -= np.expand_dims(np.sum(u * v, axis=1), -1) * u
        v = v / np.expand_dims(np.linalg.norm(v, axis=1) + 1e-9, -1)

        # Last rotation vector
        w = np.cross(u, v)
        R = np.stack((u, v, w), axis=-1)

        # Scale kernels
        original_kernel = radius * np.expand_dims(original_kernel, 0)

        # Rotate kernels
        kernels = np.matmul(original_kernel, R)

        # Add a small noise
        kernels = kernels
        kernels = kernels + np.random.normal(scale=radius * 0.01, size=kernels.shape)

    return kernels

def add_ones(query_points, x, add_one):
    if add_one:
        ones = torch.ones(query_points.shape[0], dtype=torch.float).unsqueeze(-1).to(query_points.device)
        if x is not None:
            x = torch.cat([ones.to(x.dtype), x], dim=-1)
        else:
            x = ones
    return x

def gather(x, idx, method=2):
    """
    https://github.com/pytorch/pytorch/issues/15245
    implementation of a custom gather operation for faster backwards.
    :param x: input with shape [N, D_1, ... D_d]
    :param idx: indexing with shape [n_1, ..., n_m]
    :param method: Choice of the method
    :return: x[idx] with shape [n_1, ..., n_m, D_1, ... D_d]
    """
    idx[idx == -1] = x.shape[0] - 1  # Shadow point
    if method == 0:
        return x[idx]
    elif method == 1:
        x = x.unsqueeze(1)
        x = x.expand((-1, idx.shape[-1], -1))
        idx = idx.unsqueeze(2)
        idx = idx.expand((-1, -1, x.shape[-1]))
        return x.gather(0, idx)
    elif method == 2:
        for i, ni in enumerate(idx.size()[1:]):
            x = x.unsqueeze(i + 1)
            new_s = list(x.size())
            new_s[i + 1] = ni
            x = x.expand(new_s)
        n = len(idx.size())
        for i, di in enumerate(x.size()[n:]):
            idx = idx.unsqueeze(i + n)
            new_s = list(idx.size())
            new_s[i + n] = di
            idx = idx.expand(new_s)
        return x.gather(0, idx)
    else:
        raise ValueError("Unkown method")

def KPConv_ops(
    query_points,
    support_points,
    neighbors_indices,
    features,
    K_points,
    K_values,
    KP_extent,
    KP_influence,
    aggregation_mode,
):
    """
    This function creates a graph of operations to define Kernel Point Convolution in tensorflow. See KPConv function
    above for a description of each parameter
    :param query_points: float32[n_points, dim] - input query points (center of neighborhoods)
    :param support_points: float32[n0_points, dim] - input support points (from which neighbors are taken)
    :param neighbors_indices: int32[n_points, n_neighbors] - indices of neighbors of each point
    :param features: float32[n0_points, in_fdim] - input features
    :param K_values: float32[n_kpoints, in_fdim, out_fdim] - weights of the kernel
    :param fixed: string in ('none', 'center' or 'verticals') - fix position of certain kernel points
    :param KP_extent: float32 - influence radius of each kernel point
    :param KP_influence: string in ('constant', 'linear', 'gaussian') - influence function of the kernel points
    :param aggregation_mode: string in ('closest', 'sum') - whether to sum influences, or only keep the closest
    :return:                    [n_points, out_fdim]
    """

    # Get variables
    int(K_points.shape[0])

    # Add a fake point in the last row for shadow neighbors
    shadow_point = torch.ones_like(support_points[:1, :]) * 1e6
    support_points = torch.cat([support_points, shadow_point], dim=0)

    # Get neighbor points [n_points, n_neighbors, dim]
    neighbors = gather(support_points, neighbors_indices)

    # Center every neighborhood
    neighbors = neighbors - query_points.unsqueeze(1)

    # Get all difference matrices [n_points, n_neighbors, n_kpoints, dim]
    neighbors.unsqueeze_(2)
    differences = neighbors - K_points

    # Get the square distances [n_points, n_neighbors, n_kpoints]
    sq_distances = torch.sum(differences**2, dim=3)

    # Get Kernel point influences [n_points, n_kpoints, n_neighbors]
    if KP_influence == "constant":
        # Every point get an influence of 1.
        all_weights = torch.ones_like(sq_distances)
        all_weights = all_weights.transpose(2, 1)

    elif KP_influence == "linear":
        # Influence decrease linearly with the distance, and get to zero when d = KP_extent.
        all_weights = torch.clamp(1 - torch.sqrt(sq_distances) / KP_extent, min=0.0)
        all_weights = all_weights.transpose(2, 1)

    elif KP_influence == "gaussian":
        # Influence in gaussian of the distance.
        sigma = KP_extent * 0.3
        all_weights = radius_gaussian(sq_distances, sigma)
        all_weights = all_weights.transpose(2, 1)
    else:
        raise ValueError("Unknown influence function type (config.KP_influence)")

    # In case of closest mode, only the closest KP can influence each point
    if aggregation_mode == "closest":
        neighbors_1nn = torch.argmin(sq_distances, dim=-1)
        all_weights *= torch.transpose(torch.nn.functional.one_hot(neighbors_1nn, K_points.shape[0]), 1, 2)

    elif aggregation_mode != "sum":
        raise ValueError("Unknown convolution mode. Should be 'closest' or 'sum'")

    features = torch.cat([features, torch.zeros_like(features[:1, :])], dim=0)

    # Get the features of each neighborhood [n_points, n_neighbors, in_fdim]
    neighborhood_features = gather(features, neighbors_indices)

    # Apply distance weights [n_points, n_kpoints, in_fdim]
    weighted_features = torch.matmul(all_weights, neighborhood_features)

    # Apply network weights [n_kpoints, n_points, out_fdim]
    weighted_features = weighted_features.permute(1, 0, 2)
    kernel_outputs = torch.matmul(weighted_features, K_values)

    # Convolution sum to get [n_points, out_fdim]
    output_features = torch.sum(kernel_outputs, dim=0)

    return output_features


def KPConv_deform_ops(
    query_points,
    support_points,
    neighbors_indices,
    features,
    K_points,
    offsets,
    modulations,
    K_values,
    KP_extent,
    KP_influence,
    aggregation_mode,
):
    """
    This function creates a graph of operations to define Deformable Kernel Point Convolution in tensorflow. See
    KPConv_deformable function above for a description of each parameter
    :param query_points:        [n_points, dim]
    :param support_points:      [n0_points, dim]
    :param neighbors_indices:   [n_points, n_neighbors]
    :param features:            [n0_points, in_fdim]
    :param K_points:            [n_kpoints, dim]
    :param offsets:             [n_points, n_kpoints, dim]
    :param modulations:         [n_points, n_kpoints] or None
    :param K_values:            [n_kpoints, in_fdim, out_fdim]
    :param KP_extent:           float32
    :param KP_influence:        string
    :param aggregation_mode:    string in ('closest', 'sum') - whether to sum influences, or only keep the closest

    :return features, square_distances, deformed_K_points
    """

    # Get variables
    n_kp = int(K_points.shape[0])
    shadow_ind = support_points.shape[0]

    # Add a fake point in the last row for shadow neighbors
    shadow_point = torch.ones_like(support_points[:1, :]) * 1e6
    support_points = torch.cat([support_points, shadow_point], axis=0)

    # Get neighbor points [n_points, n_neighbors, dim]
    neighbors = support_points[neighbors_indices]

    # Center every neighborhood
    neighbors = neighbors - query_points.unsqueeze(1)

    # Apply offsets to kernel points [n_points, n_kpoints, dim]
    deformed_K_points = torch.add(offsets, K_points)

    # Get all difference matrices [n_points, n_neighbors, n_kpoints, dim]
    neighbors = neighbors.unsqueeze(2)
    neighbors = neighbors.repeat([1, 1, n_kp, 1])
    differences = neighbors - deformed_K_points.unsqueeze(1)

    # Get the square distances [n_points, n_neighbors, n_kpoints]
    sq_distances = torch.sum(differences**2, axis=3)

    # Boolean of the neighbors in range of a kernel point [n_points, n_neighbors]
    in_range = (sq_distances < KP_extent**2).any(2).to(torch.long)

    # New value of max neighbors
    new_max_neighb = torch.max(torch.sum(in_range, axis=1))
    # print(new_max_neighb)

    # For each row of neighbors, indices of the ones that are in range [n_points, new_max_neighb]
    new_neighb_bool, new_neighb_inds = torch.topk(in_range, k=new_max_neighb)

    # Gather new neighbor indices [n_points, new_max_neighb]
    new_neighbors_indices = neighbors_indices.gather(1, new_neighb_inds)

    # Gather new distances to KP [n_points, new_max_neighb, n_kpoints]
    new_neighb_inds_sq = new_neighb_inds.unsqueeze(-1)
    new_sq_distances = sq_distances.gather(1, new_neighb_inds_sq.repeat((1, 1, sq_distances.shape[-1])))

    # New shadow neighbors have to point to the last shadow point
    new_neighbors_indices *= new_neighb_bool
    new_neighbors_indices += (1 - new_neighb_bool) * shadow_ind

    # Get Kernel point influences [n_points, n_kpoints, n_neighbors]
    if KP_influence == "constant":
        # Every point get an influence of 1.
        all_weights = (new_sq_distances < KP_extent**2).to(torch.float32)
        all_weights = all_weights.permute(0, 2, 1)

    elif KP_influence == "linear":
        # Influence decrease linearly with the distance, and get to zero when d = KP_extent.
        all_weights = torch.relu(1 - torch.sqrt(new_sq_distances) / KP_extent)
        all_weights = all_weights.permute(0, 2, 1)

    elif KP_influence == "gaussian":
        # Influence in gaussian of the distance.
        sigma = KP_extent * 0.3
        all_weights = radius_gaussian(new_sq_distances, sigma)
        all_weights = all_weights.permute(0, 2, 1)
    else:
        raise ValueError("Unknown influence function type (config.KP_influence)")

    # In case of closest mode, only the closest KP can influence each point
    if aggregation_mode == "closest":
        neighbors_1nn = torch.argmin(new_sq_distances, axis=2, output_type=torch.long)
        all_weights *= torch.zeros_like(all_weights, dtype=torch.float32).scatter_(1, neighbors_1nn, 1)

    elif aggregation_mode != "sum":
        raise ValueError("Unknown convolution mode. Should be 'closest' or 'sum'")

    features = torch.cat([features, torch.zeros_like(features[:1, :])], axis=0)

    # Get the features of each neighborhood [n_points, new_max_neighb, in_fdim]
    neighborhood_features = features[new_neighbors_indices]

    # Apply distance weights [n_points, n_kpoints, in_fdim]
    # print(all_weights.shape, neighborhood_features.shape)
    weighted_features = torch.matmul(all_weights, neighborhood_features)

    # Apply modulations
    if modulations is not None:
        weighted_features *= modulations.unsqueeze(2)

    # Apply network weights [n_kpoints, n_points, out_fdim]
    weighted_features = weighted_features.permute(1, 0, 2)
    kernel_outputs = torch.matmul(weighted_features, K_values)

    # Convolution sum [n_points, out_fdim]
    output_features = torch.sum(kernel_outputs, axis=0)

    # we need regularization
    return output_features, sq_distances, deformed_K_points


class KPConvLayer(nn.Module):
    """
    apply the kernel point convolution on a point cloud
    NB : it is the original version of KPConv, it is not the message passing version
    attributes:
    num_inputs : dimension of the input feature
    num_outputs : dimension of the output feature
    point_influence: influence distance of a single point (sigma * grid_size)
    n_kernel_points=15
    fixed="center"
    KP_influence="linear"
    aggregation_mode="sum"
    dimension=3
    """

    _INFLUENCE_TO_RADIUS = 1.5

    def __init__(
        self,
        num_inputs,
        num_outputs,
        point_influence,
        n_kernel_points=15,
        fixed="center",
        KP_influence="linear",
        aggregation_mode="sum",
        dimension=3,
        add_one=False,
        **kwargs
    ):
        super(KPConvLayer, self).__init__()
        self.kernel_radius = self._INFLUENCE_TO_RADIUS * point_influence
        self.point_influence = point_influence
        self.add_one = add_one
        self.num_inputs = num_inputs + self.add_one * 1
        self.num_outputs = num_outputs

        self.KP_influence = KP_influence
        self.n_kernel_points = n_kernel_points
        self.aggregation_mode = aggregation_mode

        # Initial kernel extent for this layer
        K_points_numpy = load_kernels(
            self.kernel_radius,
            n_kernel_points,
            num_kernels=1,
            dimension=dimension,
            fixed=fixed,
        )

        self.K_points = nn.Parameter(
            torch.from_numpy(K_points_numpy.reshape((n_kernel_points, dimension))).to(torch.float),
            requires_grad=False,
        )

        weights = torch.empty([n_kernel_points, self.num_inputs, num_outputs], dtype=torch.float)
        torch.nn.init.xavier_normal_(weights)
        self.weight = nn.Parameter(weights)

    def forward(self, query_points, support_points, neighbors, x):
        """
        - query_points(torch Tensor): query of size N x 3
        - support_points(torch Tensor): support points of size N0 x 3
        - neighbors(torch Tensor): neighbors of size N x M
        - features : feature of size N0 x d (d is the number of inputs)
        """
        x = add_ones(support_points, x, self.add_one)

        new_feat = KPConv_ops(
            query_points,
            support_points,
            neighbors,
            x,
            self.K_points,
            self.weight,
            self.point_influence,
            self.KP_influence,
            self.aggregation_mode,
        )
        return new_feat


class KPConvDeformableLayer(nn.Module):
    """
    apply the deformable kernel point convolution on a point cloud
    NB : it is the original version of KPConv, it is not the message passing version
    attributes:
    num_inputs : dimension of the input feature
    num_outputs : dimension of the output feature
    point_influence: influence distance of a single point (sigma * grid_size)
    n_kernel_points=15
    fixed="center"
    KP_influence="linear"
    aggregation_mode="sum"
    dimension=3
    modulated = False :   If deformable conv should be modulated
    """

    PERMISSIVE_LOSS_KEY = "permissive_loss"
    FITTING_LOSS_KEY = "fitting_loss"
    REPULSION_LOSS_KEY = "repulsion_loss"

    _INFLUENCE_TO_RADIUS = 1.5

    def __init__(
        self,
        num_inputs,
        num_outputs,
        point_influence,
        n_kernel_points=15,
        fixed="center",
        KP_influence="linear",
        aggregation_mode="sum",
        dimension=3,
        modulated=False,
        loss_mode="fitting",
        add_one=False,
        **kwargs
    ):
        super(KPConvDeformableLayer, self).__init__()
        self.kernel_radius = self._INFLUENCE_TO_RADIUS * point_influence
        self.point_influence = point_influence
        self.add_one = add_one
        self.num_inputs = num_inputs + self.add_one * 1
        self.num_outputs = num_outputs

        self.KP_influence = KP_influence
        self.n_kernel_points = n_kernel_points
        self.aggregation_mode = aggregation_mode
        self.modulated = modulated
        self.internal_losses = {self.PERMISSIVE_LOSS_KEY: 0.0, self.FITTING_LOSS_KEY: 0.0, self.REPULSION_LOSS_KEY: 0.0}
        self.loss_mode = loss_mode

        # Initial kernel extent for this layer
        K_points_numpy = load_kernels(
            self.kernel_radius,
            n_kernel_points,
            num_kernels=1,
            dimension=dimension,
            fixed=fixed,
        )
        self.K_points = nn.Parameter(
            torch.from_numpy(K_points_numpy.reshape((n_kernel_points, dimension))).to(torch.float),
            requires_grad=False,
        )

        # Create independant weight for the first convolution and a bias term as no batch normalization happen
        if modulated:
            offset_dim = (dimension + 1) * self.n_kernel_points
        else:
            offset_dim = dimension * self.n_kernel_points
        offset_weights = torch.empty([n_kernel_points, self.num_inputs, offset_dim], dtype=torch.float)
        torch.nn.init.xavier_normal_(offset_weights)
        self.offset_weights = nn.Parameter(offset_weights)
        self.offset_bias = nn.Parameter(torch.zeros(offset_dim, dtype=torch.float))

        # Main deformable weights
        weights = torch.empty([n_kernel_points, self.num_inputs, num_outputs], dtype=torch.float)
        torch.nn.init.xavier_normal_(weights)
        self.weight = nn.Parameter(weights)

    def forward(self, query_points, support_points, neighbors, x):
        """
        - query_points(torch Tensor): query of size N x 3
        - support_points(torch Tensor): support points of size N0 x 3
        - neighbors(torch Tensor): neighbors of size N x M
        - features : feature of size N0 x d (d is the number of inputs)
        """

        x = add_ones(support_points, x, self.add_one)

        offset_feat = (
            KPConv_ops(
                query_points,
                support_points,
                neighbors,
                x,
                self.K_points,
                self.offset_weights,
                self.point_influence,
                self.KP_influence,
                self.aggregation_mode,
            )
            + self.offset_bias
        )
        points_dim = query_points.shape[-1]
        if self.modulated:
            # Get offset (in normalized scale) from features
            offsets = offset_feat[:, : points_dim * self.n_kernel_points]
            offsets = offsets.reshape((-1, self.n_kernel_points, points_dim))

            # Get modulations
            modulations = 2 * torch.nn.functional.sigmoid(offset_feat[:, points_dim * self.n_kernel_points :])
        else:
            # Get offset (in normalized scale) from features
            offsets = offset_feat.reshape((-1, self.n_kernel_points, points_dim))
            # No modulations
            modulations = None
        offsets *= self.point_influence

        # Apply deformable kernel
        new_feat, sq_distances, K_points_deformed = KPConv_deform_ops(
            query_points,
            support_points,
            neighbors,
            x,
            self.K_points,
            offsets,
            modulations,
            self.weight,
            self.point_influence,
            self.KP_influence,
            self.aggregation_mode,
        )

        if self.loss_mode == "fitting":
            self.internal_losses[self.FITTING_LOSS_KEY] = fitting_loss(sq_distances, self.kernel_radius)
            self.internal_losses[self.REPULSION_LOSS_KEY] = repulsion_loss(K_points_deformed, self.point_influence)
        elif self.loss_mode == "permissive":
            self.internal_losses[self.PERMISSIVE_LOSS_KEY] = permissive_loss(K_points_deformed, self.kernel_radius)
        else:
            raise NotImplementedError(
                "Loss mode %s not recognised. Only permissive and fitting are valid" % self.loss_mode
            )
        return new_feat


@MODELS.register_module()
class KPConvSimpleBlock(nn.Module):
    """
    simple layer with KPConv convolution -> activation -> BN
    we can perform a stride version (just change the query and the neighbors)
    """

    CONV_TYPE = ConvolutionFormat.PARTIAL_DENSE.value
    DEFORMABLE_DENSITY = 5.0
    RIGID_DENSITY = 2.5

    def __init__(
        self,
        in_channels,
        KP_extent,
        sigma=1.0,
        activation=torch.nn.LeakyReLU(negative_slope=0.1),
        bn_momentum=0.02,
        bn=nn.BatchNorm1d,
        deformable=False,
        add_one=False,
        grouper=None,
        **kwargs,
    ):
        super(KPConvSimpleBlock, self).__init__()
        #assert len(down_conv_nn) == 2
        #num_inputs, num_outputs = down_conv_nn
        if deformable:
            density_parameter = self.DEFORMABLE_DENSITY
            self.kp_conv = KPConvDeformableLayer(
                in_channels, in_channels, point_influence=KP_extent * sigma, add_one=add_one, **kwargs
            )
        else:
            density_parameter = self.RIGID_DENSITY
            self.kp_conv = KPConvLayer(
                in_channels, in_channels, point_influence=KP_extent * sigma, add_one=add_one, **kwargs
            )

        #self.neighbour_finder = RadiusNeighbourFinder(search_radius, max_num_neighbors, conv_type=self.CONV_TYPE)

        self.grouper = MODELS.build(grouper)

        if bn:
            self.bn = bn(in_channels, momentum=bn_momentum)
        else:
            self.bn = None
        self.activation = activation


    def forward(self, pxo, **kwargs):
        p, x, o = pxo
        _, idx_neighboors = self.grouper(
            dict(
                source_p = p,
                target_p = p,
                source_f = x,
                source_o = o,
                target_o = o,
                with_xyz = True,
            )
        )

        x = self.kp_conv(
            p,
            p,
            idx_neighboors.long(),
            x,
        )
        if self.bn:
            x = self.bn(x)
        x = self.activation(x)

        return p, x, o



@MODELS.register_module()
class KPConvResnetBBlock(nn.Module):
    """Resnet block with optional bottleneck activated by default
    Arguments:
        down_conv_nn (len of 2 or 3) :
                        sizes of input, intermediate, output.
                        If length == 2 then intermediate =  num_outputs // 4
        radius : radius of the conv kernel
        sigma :
        density_parameter : density parameter for the kernel
        max_num_neighbors : maximum number of neighboors for the neighboor search
        activation : activation function
        has_bottleneck: wether to use the bottleneck or not
        bn_momentum
        bn : batch norm (can be None -> no batch norm)
    """

    CONV_TYPE = ConvolutionFormat.PARTIAL_DENSE.value

    def __init__(
        self,
        in_channels,
        KP_extent,
        sigma=1,
        activation=torch.nn.LeakyReLU(negative_slope=0.1),
        has_bottleneck=True,
        mid_channels=None,
        bn_momentum=0.02,
        bn=nn.BatchNorm1d,
        deformable=False,
        add_one=False,
        grouper=None,
        **kwargs,
    ):
        super(KPConvResnetBBlock, self).__init__()
        #assert len(down_conv_nn) == 2 or len(down_conv_nn) == 3, "down_conv_nn should be of size 2 or 3"
        if mid_channels is None:
            d_2 = in_channels // 4
        else:
            d_2 = mid_channels

        self.has_bottleneck = has_bottleneck

        # Main branch
        if self.has_bottleneck:
            kp_size = [d_2, d_2]
        else:
            kp_size = [in_channels, in_channels]

        self.kp_conv = KPConvSimpleBlock(
            in_channels = kp_size[0],
            out_channels = kp_size[1],
            KP_extent=KP_extent,
            sigma=sigma,
            activation=activation,
            bn_momentum=bn_momentum,
            bn=bn,
            deformable=deformable,
            add_one=add_one,
            grouper=grouper,
            **kwargs,
        )

        if self.has_bottleneck:
            if bn:
                self.unary_1 = torch.nn.Sequential(
                    nn.Linear(in_channels, d_2, bias=False), bn(d_2, momentum=bn_momentum), activation
                )
                self.unary_2 = torch.nn.Sequential(
                    nn.Linear(d_2, in_channels, bias=False), bn(in_channels, momentum=bn_momentum), activation
                )
            else:
                self.unary_1 = torch.nn.Sequential(nn.Linear(in_channels, d_2, bias=False), activation)
                self.unary_2 = torch.nn.Sequential(nn.Linear(d_2, in_channels, bias=False), activation)

        # Shortcut
        if in_channels != in_channels:
            if bn:
                self.shortcut_op = torch.nn.Sequential(
                    nn.Linear(in_channels, in_channels, bias=False), bn(in_channels, momentum=bn_momentum)
                )
            else:
                self.shortcut_op = nn.Linear(in_channels, in_channels, bias=False)
        else:
            self.shortcut_op = torch.nn.Identity()

        # Final activation
        self.activation = activation

    def forward(self, pxo, **kwargs):
        """
        data: x, pos, batch_idx and idx_neighbour when the neighboors of each point in pos have already been computed
        """
        # Main branch
        p, x, o = pxo
        shortcut_x = x
        if self.has_bottleneck:
            x = self.unary_1(x)
        p, x, o = self.kp_conv([p, x, o])
        if self.has_bottleneck:
            x = self.unary_2(x)

        shortcut = self.shortcut_op(shortcut_x)
        x += shortcut
        return p, x, o