import torch
import torch.nn as nn

from .actor_critic import get_activation


class ActorCriticParkourStudent(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_actions,
        num_teacher_actor_obs,
        num_history_obs,
        num_latent_dims,
        student_depth_shape=(2, 58, 87),
        proprio_history_frames=10,
        proprio_history_hidden_dims=[128, 64],
        depth_encoder_hidden_dims=[128, 64],
        student_latent_hidden_dims=[256, 128],
        actor_hidden_dims=[512, 256, 128],
        activation="elu",
        clip_actions=100.0,
        **kwargs,
    ):
        if kwargs:
            raise TypeError(
                "ActorCriticParkourStudent received unexpected arguments: "
                + str(sorted(kwargs.keys()))
            )
        super().__init__()

        activation_layer = get_activation(activation)
        self.num_obs = int(num_actor_obs)
        self.num_teacher_actor_obs = int(num_teacher_actor_obs)
        self.num_history_obs = int(num_history_obs)
        self.num_latent_dims = int(num_latent_dims)
        self.student_depth_shape = tuple(student_depth_shape)
        self.proprio_history_frames = int(proprio_history_frames)

        if self.num_history_obs != self.num_obs * self.proprio_history_frames:
            raise ValueError(
                "ActorCriticParkourStudent expects obs_history to be "
                f"{self.num_obs} * {self.proprio_history_frames}, got {self.num_history_obs}."
            )

        history_latent_dim = proprio_history_hidden_dims[-1]
        depth_latent_dim = depth_encoder_hidden_dims[-1]

        self.proprio_history_encoder = self._build_history_encoder(
            num_obs=self.num_obs,
            num_frames=self.proprio_history_frames,
            hidden_dims=proprio_history_hidden_dims,
            activation=activation_layer,
        )
        self.depth_encoder = self._build_depth_encoder(
            depth_shape=self.student_depth_shape,
            hidden_dims=depth_encoder_hidden_dims,
            activation=activation_layer,
        )
        self.student_latent_encoder = self._build_mlp(
            input_dim=self.num_obs + history_latent_dim + depth_latent_dim,
            hidden_dims=student_latent_hidden_dims,
            output_dim=self.num_latent_dims,
            activation=activation_layer,
        )
        self.actor = self._build_mlp(
            input_dim=self.num_obs + self.num_latent_dims,
            hidden_dims=actor_hidden_dims,
            output_dim=num_actions,
            activation=activation_layer,
        )
        self.actor.add_module(
            "clip_actions",
            nn.Hardtanh(min_val=-clip_actions, max_val=clip_actions),
        )

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
    def _build_depth_encoder(depth_shape, hidden_dims, activation):
        in_channels, height, width = depth_shape
        conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2),
            type(activation)(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            type(activation)(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            type(activation)(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, height, width)
            flat_size = conv_layers(dummy).shape[1]
        projection_layers = []
        prev_dim = flat_size
        for hidden_dim in hidden_dims:
            projection_layers.append(nn.Linear(prev_dim, hidden_dim))
            projection_layers.append(type(activation)())
            prev_dim = hidden_dim
        return nn.Sequential(conv_layers, *projection_layers)

    @staticmethod
    def _build_history_encoder(num_obs, num_frames, hidden_dims, activation):
        conv_stack = nn.Sequential(
            nn.Conv1d(num_obs, 64, kernel_size=3, padding=1),
            type(activation)(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            type(activation)(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, num_obs, num_frames)
            flat_size = conv_stack(dummy).shape[1]
        projection_layers = []
        prev_dim = flat_size
        for hidden_dim in hidden_dims:
            projection_layers.append(nn.Linear(prev_dim, hidden_dim))
            projection_layers.append(type(activation)())
            prev_dim = hidden_dim
        return nn.Sequential(conv_stack, *projection_layers)

    def _reshape_obs_history(self, obs_history):
        return obs_history.view(-1, self.proprio_history_frames, self.num_obs).transpose(1, 2)

    def infer_student_latent(self, observations, student_depth, obs_history):
        depth_latent = self.depth_encoder(student_depth)
        history_latent = self.proprio_history_encoder(self._reshape_obs_history(obs_history))
        encoder_input = torch.cat((observations, history_latent, depth_latent), dim=-1)
        return self.student_latent_encoder(encoder_input)

    def actor_from_latent(self, observations, student_latent):
        return self.actor(torch.cat((observations, student_latent), dim=-1))

    def act(self, observations, student_depth, obs_history):
        return self.actor_from_latent(
            observations,
            self.infer_student_latent(observations, student_depth, obs_history),
        )

    def act_inference(self, observations, student_depth, obs_history):
        return self.act(observations, student_depth, obs_history)

    def load_teacher_actor_weights(self, teacher_state_dict):
        actor_state = {
            key.removeprefix("actor."): value
            for key, value in teacher_state_dict.items()
            if key.startswith("actor.")
        }
        missing, unexpected = self.actor.load_state_dict(actor_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Teacher actor initialization mismatch. "
                f"Missing keys: {missing}. Unexpected keys: {unexpected}."
            )
