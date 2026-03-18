import torch
import torch.nn as nn
from torch.distributions import Normal

from .actor_critic import get_activation


class ActorCriticParkourStudent(nn.Module):
    """Actor-critic for parkour student training via depth-based distillation.

    During PPO rollouts the actor uses the scandot_encoder (privileged) latent.
    A separate depth_encoder CNN is trained via supervised MSE to match the
    scandot_encoder output so that at deployment only depth images are needed.
    """

    is_recurrent = False
    history_encoder_type = "CNN"

    def __init__(
        self,
        num_actor_obs,
        num_actions,
        num_privilege_encoder_input,
        num_history_encoder_input,  # = num_depth_obs (flattened), unused for CNN sizing
        num_latent_dims,
        num_critic_obs,
        scandot_encoder_hidden_dims=[128, 64, 32],
        depth_image_shape=(1, 58, 87),
        depth_encoder_hidden_dim=32,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[1024, 512, 256],
        activation="elu",
        init_noise_std=1.0,
        clip_actions=100.0,
        teacher_checkpoint=None,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticParkourStudent.__init__ got unexpected arguments, "
                "which will be ignored: " + str(list(kwargs.keys()))
            )
        super().__init__()

        activation_layer = get_activation(activation)
        self.depth_image_shape = tuple(depth_image_shape)
        latent_dim = num_latent_dims

        # --- Scandot encoder (same architecture as teacher) ---
        self.scandot_encoder = self._build_mlp(
            input_dim=num_privilege_encoder_input,
            hidden_dims=scandot_encoder_hidden_dims,
            output_dim=None,
            activation=activation_layer,
            final_activation=True,
        )
        scandot_latent_dim = scandot_encoder_hidden_dims[-1]
        if scandot_latent_dim != latent_dim:
            raise ValueError(
                f"scandot_encoder output dim ({scandot_latent_dim}) must match "
                f"num_latent_dims ({latent_dim})"
            )

        # --- Depth encoder (CNN) ---
        C, H, W = self.depth_image_shape
        self.depth_encoder = self._build_depth_cnn(
            in_channels=C, height=H, width=W,
            latent_dim=latent_dim,
            activation=activation_layer,
        )

        # --- Actor ---
        self.actor = self._build_mlp(
            input_dim=num_actor_obs + latent_dim,
            hidden_dims=actor_hidden_dims,
            output_dim=num_actions,
            activation=activation_layer,
        )
        self.actor.add_module(
            "clip_actions",
            nn.Hardtanh(min_val=-clip_actions, max_val=clip_actions),
        )

        # --- Critic ---
        self.critic = self._build_mlp(
            input_dim=num_critic_obs,
            hidden_dims=critic_hidden_dims,
            output_dim=1,
            activation=activation_layer,
        )

        print(f"Scandot Encoder MLP: {self.scandot_encoder}")
        print(f"Depth Encoder CNN: {self.depth_encoder}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

    # ---------- Aliases for PPO_TS compatibility ----------

    @property
    def privilege_encoder(self):
        return self.scandot_encoder

    @property
    def history_encoder(self):
        return self.depth_encoder

    # ---------- Network builders ----------

    @staticmethod
    def _build_mlp(input_dim, hidden_dims, output_dim, activation, final_activation=False):
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(type(activation)())
            prev_dim = hidden_dim
        if output_dim is not None:
            layers.append(nn.Linear(prev_dim, output_dim))
            if final_activation:
                layers.append(type(activation)())
        elif not final_activation and layers:
            layers.pop()
        return nn.Sequential(*layers)

    @staticmethod
    def _build_depth_cnn(in_channels, height, width, latent_dim, activation):
        conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2),
            type(activation)(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            type(activation)(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            type(activation)(),
            nn.Flatten(),
        )
        # Compute flattened size after convolutions
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, height, width)
            flat_size = conv_layers(dummy).shape[1]
        return nn.Sequential(
            conv_layers,
            nn.Linear(flat_size, 128),
            type(activation)(),
            nn.Linear(128, latent_dim),
        )

    # ---------- Distribution interface ----------

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    # ---------- Actor forward paths ----------

    def update_distribution(self, observations, privilege_observations):
        latent = self.scandot_encoder(privilege_observations)
        mean = self.actor(torch.cat((observations, latent), dim=-1))
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, privilege_observations, **kwargs):
        self.update_distribution(observations, privilege_observations)
        return self.distribution.sample()

    def act_student(self, observations, depth_obs, **kwargs):
        depth_image = depth_obs.view(-1, *self.depth_image_shape)
        latent = self.depth_encoder(depth_image)
        return self.actor(torch.cat((observations, latent), dim=-1))

    def act_inference(self, observations, depth_obs, **kwargs):
        return self.act_student(observations, depth_obs, **kwargs)

    # ---------- Critic ----------

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)

    # ---------- Teacher weight loading ----------

    def load_teacher_weights(self, teacher_state_dict):
        """Load matching weights from a trained ActorCriticParkour teacher checkpoint."""
        own = self.state_dict()
        loaded = 0
        for key, value in teacher_state_dict.items():
            if key in own and own[key].shape == value.shape:
                own[key] = value
                loaded += 1
        self.load_state_dict(own)
        print(f"Loaded {loaded} teacher weight tensors into student.")
