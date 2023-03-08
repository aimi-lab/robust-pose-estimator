from lietorch import SE3, LieGroupParameter
from core.ddn.ddn.pytorch.node import *
from typing import Tuple



class DeclarativeNodeLie(AbstractDeclarativeNode):
    def __init__(self, eps=1e-3, dbg=False):
        super(DeclarativeNodeLie, self).__init__(eps=eps)
        self.dbg = dbg

    # we re-implement the gradient function with more error-handling to catch failed optimization runs
    def gradient(self, *xs:Tuple[torch.tensor], y:Tuple[torch.tensor]=None, v:Tuple[torch.tensor]=None, ctx=None):
        """Computes the vector--Jacobian product, that is, the gradient of the
        loss function with respect to the problem parameters. The returned
        gradient is a tuple of batched Torch tensors. Can be overridden by the
        derived class to provide a more efficient implementation.

        Arguments:
            xs: ((b, ...), ...) tuple of Torch tensors,
                tuple of batches of input tensors

            y: ((b, ...), ...) tuple of Torch tensors,
                batch of minima of the objective function

            v: ((b, ...), ...) tuple of Torch tensors,
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
        v = torch.cat([a.reshape(self.b, -1,1) for a in v], dim=1)
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

    def _gradient_init(self, xs, y, v, ctx):
        self.b = y[0].size(0)
        self.m = sum([y_i.reshape(self.b, -1).size(-1)for y_i in y])

        # Split each input x into a tuple of n//chunk_size tensors of size (b, chunk_size):
        # Required since gradients can only be computed wrt individual
        # tensors, not slices of a tensor. See:
        # https://discuss.pytorch.org/t/how-to-calculate-gradients-wrt-one-of-inputs/24407
        xs_split, xs_sizes, self.n = self._split_inputs(xs)
        xs = self._cat_inputs(xs_split, xs_sizes)

        return xs, xs_split, xs_sizes, y, v, ctx

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
        f = torch.enable_grad()(self.objective)(*xs, y=y, backward=True)  # b

        # Compute partial derivative of f wrt y at (xs,y):
        fY = grad(f, y, grad_outputs=torch.ones_like(f), create_graph=True)
        with torch.enable_grad():
            fY = torch.cat([fy.reshape(self.b, -1) for fy in fY], dim=-1)

        # Compute second-order partial derivative of f wrt y at (xs,y):
        # fYY = self._batch_jacobian_fast(fY, y) autograd.Function not yet supported by functorch
        fYY = self._batch_jacobian(fY, y)  # bxmxm

        fYY = fYY.detach()

        # Create function that returns generator expression for fXY given input:
        fXY = lambda x: (fXiY.detach()
                         if fXiY is not None else torch.zeros_like(fY).unsqueeze(-1)
                         for fXiY in (self._batch_jacobian(fY, xi) for xi in x))

        return fY, fYY, fXY

    @torch.enable_grad()
    def _batch_jacobian(self, y, x, create_graph=False):
        """Compute Jacobian of y with respect to x and reduce over batch
        dimension.

        Arguments:
            y: (b, m1, m2, ...) Torch tensor,
                batch of output tensors

            x: (b, n1, n2, ...) Torch tensor,
                batch of input tensors

            create_graph: Boolean
                if True, graph of the derivative will be constructed,
                allowing the computation of higher order derivative products

        Return Values:
            jacobian: (b, m, n) Torch tensor,
                batch of Jacobian matrices, collecting the partial derivatives
                of y with respect to x
                m = product(m_i)
                n = product(n_i)

        Assumption:
            If x is not in graph for y[:, 0], then x is not in the graph for
            y[:, i], for all i
        """

        y = y.reshape(self.b, -1) # bxm
        m = y.size(-1)
        if isinstance(x, tuple):
            n = sum([x_i.reshape(self.b, -1).size(-1) for x_i in x])
        else:
            n = x.reshape(self.b, -1).size(-1)
        jacobian = y.new_zeros(self.b, m, n) # bxmxn
        for i in range(m):
            grad_outputs = torch.zeros_like(y, requires_grad=False) # bxm
            grad_outputs[:, i] = 1.0
            yiX = grad(y, x, grad_outputs=grad_outputs, retain_graph=True,
                create_graph=create_graph) # bxn1xn2x...
            if any([a is None for a in yiX]):  # fY is independent from y
                raise RuntimeError('fY is independent from y')
            jacobian[:, i:(i+1), :] = torch.cat([a.reshape(self.b, 1, -1) for a in yiX], dim=-1)
        return jacobian # bxmxn

    @torch.enable_grad()
    def _batch_jacobian_fast(self, y, x, create_graph=False):
        """Compute Jacobian of y with respect to x and reduce over batch
        dimension.

        Arguments:
            y: (b, m1, m2, ...) Torch tensor,
                batch of output tensors

            x: (b, n1, n2, ...) Torch tensor,
                batch of input tensors

            create_graph: Boolean
                if True, graph of the derivative will be constructed,
                allowing the computation of higher order derivative products

        Return Values:
            jacobian: (b, m, n) Torch tensor,
                batch of Jacobian matrices, collecting the partial derivatives
                of y with respect to x
                m = product(m_i)
                n = product(n_i)

        Assumption:
            If x is not in graph for y[:, 0], then x is not in the graph for
            y[:, i], for all i
        """
        from functorch import vmap, jacrev
        b, m = y.shape

        def get_vjp(v):
            return torch.autograd.grad(y, x, v)

        grad_outputs = torch.eye(m, requires_grad=False, dtype=y.dtype, device=y.device).repeat((b, 1,1))

        batch_jacobian = vmap(get_vjp, in_dims=(1))(grad_outputs)
        return batch_jacobian


class DeclarativeFunctionLie(DeclarativeFunction):
    """Lie declarative autograd function with tuple inputs and outputs.
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
        outs_embed = []
        outs_tan = []
        outs_type = []
        with torch.no_grad():
            *outs, solve_ctx = problem.solve(*inputs)
            # store outputs in embedding space for backward and in tangent space for loss computations
            for out in outs:
                if isinstance(out, LieGroupParameter):
                    outs_embed.append(out.group.vec().detach().float())
                    outs_tan.append(out.log().detach().float())
                    outs_type.append(LieGroupParameter)
                elif torch.is_tensor(out):
                    outs_embed.append(out.detach().float())
                    outs_tan.append(out.detach().float())  # tangent space is embedding space for vector
                    outs_type.append(torch.TensorType)
                else:
                    raise NotImplementedError

        ctx.save_for_backward(*outs_embed, *inputs)
        ctx.out_types = outs_type
        ctx.problem = problem
        ctx.solve_ctx = solve_ctx
        return tuple([out.clone() for out in outs_embed]) + tuple(outs_tan)

    @staticmethod
    def backward(ctx, *grad_outs):
        out_types = ctx.out_types
        outs = ctx.saved_tensors[:len(out_types)]
        inputs = ctx.saved_tensors[len(out_types):]
        problem = ctx.problem
        solve_ctx = ctx.solve_ctx
        outs_param = []
        for out, out_type in zip(outs, out_types):
            if out_type == LieGroupParameter:
                out = LieGroupParameter(SE3(out))
            out.requires_grad = True
            outs_param.append(out)

        inputs = tuple(inputs)
        outs_param = tuple(outs_param)
        grad_outs_tan = grad_outs[len(out_types):]  # (embedding grad outs, tanget grad outs)
        grad_inputs = problem.gradient(*inputs, y=outs_param, v=grad_outs_tan, ctx=solve_ctx)
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