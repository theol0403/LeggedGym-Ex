import torch

class ConstraintManager:
    """Handle the computation of termination probabilities based on constraints
    violations (Constraints as Terminations). refer to https://constraints-as-terminations.github.io/

    Args:
        tau (float): discount factor
        min_p (float): minimum termination probability
    """

    def __init__(self, tau=0.95, min_p=0.0):
        self.running_maxes = {}  # Polyak average of the maximum constraint violation
        self.running_mins = {}  # Polyak average of the minimum constraint violation
        self.probs = {}  # Termination probabilities for each constraint
        self._batched_probs = None
        self.tau = tau  # Discount factor
        self.min_p = min_p  # Minimum termination probability

    def reset(self):
        """Reset the termination probabilities of the constraint manager."""
        self.probs = {}
        self._batched_probs = None

    def add(self, name, constraint, max_p=0.1):
        """Add a constraint violation to the constraint manager and compute the
        associated termination probability.

        Args:
            name (string): name of the constraint
            constraint (float tensor): value of constraint violations for this constraint
            max_p (float): maximum termination probability
        """

        # First, put constraint in the form Torch.FloatTensor((num_envs, n_constraints))
        # Convert constraints violation to float if they are not
        if not torch.is_floating_point(constraint):
            constraint = constraint.float()

        # Ensure constraint is 2-dimensional even with a single element
        if constraint.ndim == 1:
            constraint = constraint.unsqueeze(1)

        # Get the maximum constraint violation for the current step
        constraint_max = constraint.amax(dim=0, keepdim=True).clamp_min(1e-6)

        # Compute polyak average of the maximum constraint violation for this constraint
        if name not in self.running_maxes:
            self.running_maxes[name] = constraint_max
        else:
            self.running_maxes[name].mul_(self.tau).add_(constraint_max, alpha=1.0 - self.tau)

        # Compute the termination probability which scales between min_p and max_p
        # with increasing constraint violation, while staying zero when inactive.
        normalized = (constraint / self.running_maxes[name]).clamp_(min=0.0, max=1.0)
        probs = torch.where(
            constraint > 0.0,
            self.min_p + normalized * (max_p - self.min_p),
            torch.zeros_like(constraint),
        )
        self.probs[name] = probs
        self._batched_probs = None

    def add_many(self, names, constraints, max_p):
        """Add multiple constraints in one batched pass.

        Args:
            names (Sequence[str]): Stable names for each constraint column.
            constraints (Tensor): Constraint values with shape ``(num_envs, num_constraints)``.
            max_p (Tensor | Sequence[float]): Maximum probability per constraint.
        """
        if not torch.is_floating_point(constraints):
            constraints = constraints.float()

        if constraints.ndim == 1:
            constraints = constraints.unsqueeze(1)

        max_p = torch.as_tensor(max_p, device=constraints.device, dtype=constraints.dtype).view(1, -1)
        if constraints.shape[1] != len(names):
            raise ValueError(
                f"Expected {len(names)} constraint columns, got {constraints.shape[1]}"
            )

        constraint_max = constraints.amax(dim=0, keepdim=True).clamp_min(1e-6)
        for idx, name in enumerate(names):
            column_max = constraint_max[:, idx : idx + 1]
            if name not in self.running_maxes:
                self.running_maxes[name] = column_max
            else:
                self.running_maxes[name].mul_(self.tau).add_(column_max, alpha=1.0 - self.tau)

        running_max = torch.cat([self.running_maxes[name] for name in names], dim=1)
        normalized = (constraints / running_max).clamp_(min=0.0, max=1.0)
        probs = torch.where(
            constraints > 0.0,
            self.min_p + normalized * (max_p - self.min_p),
            torch.zeros_like(constraints),
        )
        self._batched_probs = probs
        self.probs = {
            name: probs[:, idx : idx + 1]
            for idx, name in enumerate(names)
        }

    def get_probs(self):
        """Returns the termination probabilities due to constraint violations."""
        if self._batched_probs is not None:
            probs = self._batched_probs
        else:
            probs = torch.cat(list(self.probs.values()), dim=1)
        probs = probs.max(1).values
        return probs

    def get_str(self, names=None):
        """Get a debug string with constraints names and their average termination probabilities"""
        if names is None:
            names = list(self.probs.keys())
        txt = ""
        for name in names:
            txt += " {}: {}".format(
                name,
                str(
                    100.0 * self.probs[name].max(1).values.gt(0.0).float().mean().item()
                )[:4],
            )
            # txt += " {}: {}".format(name, str(100.0*self.probs[name].max(1).values.float().mean().item())[:4])

        return txt[1:]

    def log_all(self, episode_sums):
        """Log terminations probabilities in episode_sums with cstr_NAME key."""
        for name in list(self.probs.keys()):
            values = self.probs[name].max(1).values.gt(0.0).float()
            if "cstr_" + name not in episode_sums:
                episode_sums["cstr_" + name] = torch.zeros_like(values)
            episode_sums["cstr_" + name] += values

    def get_names(self):
        """Return a list of all constraint names."""
        return list(self.probs.keys())

    def get_vals(self):
        """Return a list of all constraint termination probabilities."""
        res = []
        for key in self.probs.keys():
            res += [100.0 * self.probs[key].max(1).values.gt(0.0).float().mean().item()]
        return res
