from lietorch import SE3, LieGroupParameter

from core.geometry.pinhole_transforms import transform, project
from core.ddn.ddn.pytorch.node import *


class DeclarativePoseHead3DNode(AbstractDeclarativeNode):
    def __init__(self, img_coordinates, lbgfs_iters=100):
        super(DeclarativePoseHead3DNode, self).__init__(eps=1e-3)
        self.img_coordinates = img_coordinates
        self.lbgfs_iters = lbgfs_iters

    def reprojection_objective(self, flow, pcl1, weights1, mask1, intrinsics, y, ret_res=False):
        """
            r2D - reprojection residuals
        """
        n, _, h, w = flow.shape
        # project 3D-pcl to image plane
        warped_pts = project(pcl1.view(n,3,-1), y, intrinsics)
        flow_off = self.img_coordinates[None, :2] + flow.view(n, 2, -1)
        # compute residuals
        residuals = torch.sum((flow_off - warped_pts)**2, dim=1)
        residuals *= weights1.view(n, -1)
        # mask out invalid residuals
        valid = (flow_off[:, 0] > 0) & (flow_off[:, 1] > 0) & (flow_off[:, 0] < w) & (flow_off[:, 1] < h)
        valid = torch.isinf(residuals) | torch.isnan(residuals) | ~valid.view(n,-1) | ~mask1.view(n,-1)

        # weight residuals by confidences
        residuals[valid] = 0.0
        loss = torch.mean(residuals, dim=1) / (h*w)  # normalize with width and height
        if ret_res:
            flow = warped_pts - self.img_coordinates[None, :2]
            return loss, residuals, flow
        return loss

    def depth_objective(self, pcl1, pcl2, weights2, mask1, mask2, y, backward=False, ret_res=False):
        """
            r3D - point-to-point 3D residuals
        """
        n, _, h, w = pcl1.shape
        # transform point cloud given the pose
        pcl1_aligned = transform(pcl1.view(n, 3, -1), y, double_backward=backward).reshape(n, 3, h, w)[:, :3]
        # compute residuals
        residuals = torch.sum((pcl1_aligned.view(n, 3, -1) - pcl2.view(n, 3, -1)) ** 2, dim=1)
        # reweighing residuals
        residuals *= weights2.view(n, -1)
        # mask out invalid residuals
        valid = mask1 & mask2
        residuals[~valid.view(n, -1)] = 0.0
        if ret_res:
            return torch.mean(residuals, dim=-1), residuals
        return torch.mean(residuals, dim=-1)

    def objective(self, *xs, y, backward=False):
        flow, pcl1, pcl2, weights1, weights2, mask1, mask2, loss_weight, intrinsics= xs
        loss3d = self.depth_objective(pcl1, pcl2, weights2, mask1, mask2, y, backward)
        loss2d = self.reprojection_objective(flow, pcl1, weights1,mask1, intrinsics, y)
        return loss_weight[:, 1]*loss2d + loss_weight[:, 0]*loss3d

    def solve(self, *xs):
        self.img_coordinates = self.img_coordinates.to(xs[0].device)
        xs = [x.detach().clone()for x in xs]
        xs = [x.double() if x.dtype==torch.float32 else x for x in xs]
        with torch.enable_grad():
            n = xs[0].shape[0]
            # Solve using LBFGS optimizer:
            y = LieGroupParameter(SE3.Identity(n, 1, device=xs[0].device, requires_grad=True, dtype=torch.float64))
            # don't use strong-wolfe with lietorch SE3, it does not converge
            optimizer = torch.optim.LBFGS([y], lr=1.0, max_iter=self.lbgfs_iters, line_search_fn=None, )

            def fun():
                optimizer.zero_grad()
                loss = self.objective(*xs, y=y).sum()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(y, 10)
                return loss
            optimizer.step(fun)
        return y.group.vec().detach().float(), y.log().detach().float(), None

    # we re-implement the gradient function with more error-handling to catch failed optimization runs
    def gradient(self, *xs, y=None, v=None, ctx=None):
        """Computes the vector--Jacobian product, that is, the gradient of the
        loss function with respect to the problem parameters. The returned
        gradient is a tuple of batched Torch tensors. Can be overridden by the
        derived class to provide a more efficient implementation.

        Arguments:
            xs: ((b, ...), ...) tuple of Torch tensors,
                tuple of batches of input tensors

            y: (b, ...) Torch tensor or None,
                batch of minima of the objective function

            v: (b, ...) Torch tensor or None,
                batch of gradients of the loss function with respect to the
                problem output J_Y(x,y)

            ctx: dictionary of contextual information used for computing the
                 gradient

        Return Values:
            gradients: ((b, ...), ...) tuple of Torch tensors or Nones,
                batch of gradients of the loss function with respect to the
                problem parameters;
                strictly, returns the vector--Jacobian products J_Y(x,y) * y'(x)
        """
        xs, xs_split, xs_sizes, y, v, ctx = self._gradient_init(xs, y, v, ctx)

        fY, fYY, fXY = self._get_objective_derivatives(xs, y)

        if not self._check_optimality_cond(fY):
            warnings.warn(
                "Non-zero objective function gradient at y:\n{}".format(
                    fY.detach().squeeze().cpu().numpy()))
            return self.zero_grad(*xs)

        # Form H:
        H = fYY
        H = 0.5 * (H + H.transpose(1, 2))  # Ensure that H is symmetric
        if self.gamma is not None:
            H += self.gamma * torch.eye(
                self.m, dtype=H.dtype, device=H.device).unsqueeze(0)

        # Solve u = -H^-1 v:
        v = v.reshape(self.b, -1, 1)
        try:
            u = self._solve_linear_system(H, -1.0 * v)  # bxmx1
        except: # torch._C._LinAlgError
            warnings.warn("linear system is not positive definite ")
            return self.zero_grad(*xs)

        u = u.squeeze(-1)  # bxm

        u[torch.isnan(u)] = 0.0  # check for nan values

        # Compute -b_i^T H^-1 v (== b_i^T u) for all i:
        gradients = []
        for x_split, x_size, n in zip(xs_split, xs_sizes, self.n):
            if isinstance(x_split[0], torch.Tensor) and x_split[0].requires_grad:
                gradient = []
                for Bi in fXY(x_split):
                    gradient.append(torch.einsum('bmc,bm->bc', (Bi, u)))
                gradient = torch.cat(gradient, dim=-1)  # bxn
                gradient[torch.isnan(gradient)] = 0.0  # nan values may occur due to zero-weights
                gradients.append(gradient.reshape(x_size))
            else:
                gradients.append(None)
        return tuple(gradients)

    def zero_grad(self, *xs):
        # set gradients to zero, so we do not perform an update
        gradients = []
        for x in xs:
            if x.requires_grad:
                gradients.append(torch.zeros_like(x, requires_grad=False))
            else:
                gradients.append(None)
        return tuple(gradients)

    def _get_objective_derivatives(self, xs, y):
        # Evaluate objective function at (xs,y):
        y = LieGroupParameter(SE3(y))
        f = torch.enable_grad()(self.objective)(*xs, y=y, backward=True)  # b

        # Compute partial derivative of f wrt y at (xs,y):
        fY = grad(f, y, grad_outputs=torch.ones_like(f), create_graph=True)[0]
        fY = torch.enable_grad()(fY.reshape)(self.b, -1)  # bxm
        if not fY.requires_grad:  # if fY is independent of y
            fY.requires_grad = True

        # Compute second-order partial derivative of f wrt y at (xs,y):
        fYY = self._batch_jacobian(fY, y)  # bxmxm
        fYY = fYY.detach() if fYY is not None else y.new_zeros(
            self.b, self.m, self.m)
        print(fYY)

        # Create function that returns generator expression for fXY given input:
        fXY = lambda x: (fXiY.detach()
                         if fXiY is not None else torch.zeros_like(fY).unsqueeze(-1)
                         for fXiY in (self._batch_jacobian(fY, xi) for xi in x))

        return fY, fYY, fXY


class DeclarativeFunctionLie(DeclarativeFunction):
    """Lie declarative autograd function.
    Backpropagation in tangent space.
    Defines the forward and backward functions. Saves all inputs and outputs,
    which may be memory-inefficient for the specific problem.

    returns unit-quaternions for inference and tangent space vector for back-propagation

    Assumptions:
    * All inputs are PyTorch tensors
    * All inputs have a single batch dimension (b, ...)
    """
    @staticmethod
    def forward(ctx, problem, *inputs):
        output_vec, output_tan, solve_ctx = torch.no_grad()(problem.solve)(*inputs)
        ctx.save_for_backward(output_vec, *inputs)
        ctx.problem = problem
        ctx.solve_ctx = solve_ctx
        return output_vec.clone(), output_tan

    @staticmethod
    def backward(ctx, grad_output_vec, grad_output_tan):
        output, *inputs = ctx.saved_tensors
        problem = ctx.problem
        solve_ctx = ctx.solve_ctx
        output.requires_grad = True
        inputs = tuple(inputs)
        grad_inputs = problem.gradient(*inputs, y=output, v=grad_output_tan,
                                       ctx=solve_ctx)
        return (None, *grad_inputs)


class DeclarativeLayerLie(DeclarativeLayer):
    """Lie declarative layer.

    Assumptions:
    * All inputs are PyTorch tensors
    * All inputs have a single batch dimension (b, ...)

    Usage:
        problem = <derived class of *DeclarativeNode>
        declarative_layer = DeclarativeLayer(problem)
        y = declarative_layer(x1, x2, ...)
    """

    def forward(self, *inputs):
        return DeclarativeFunctionLie.apply(self.problem, *inputs)