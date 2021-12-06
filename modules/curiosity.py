import torch
import torch.nn.functional as F

from CustomNeuralNetwork import CustomNeuralNetwork

MSELoss = torch.nn.MSELoss()

CrossEntropyLoss = torch.nn.CrossEntropyLoss()

class Curiosity():
    def __init__(
        self, 
        inverse_model_params = None,
        forward_model_params = None,
        embedding_size = 1,
        inv_coef = 0.1,
        beta = 1.0
    ):
        self.embedder = None
        self.inverse_model = None
        self.forward_model = None
        self.beta = beta
        self.inv_coef = inv_coef
        self.embedding_size = embedding_size

        self.init_models(inverse_model_params, forward_model_params)
    
    def init_models(self, inverse_model_params, forward_model_params):
        if inverse_model_params is None:
            if inverse_model_params is None:
                inverse_model_params = {
                "layers_info": {
                    "n_hidden_layers": 1,
                    "types": "linear",
                    "sizes": 1,
                    "hidden_activations": "relu",
                    "output_activation": "softmax",
                },
                "optimizer_info": {
                    "type": "adam",
                    "learning_rate": 0.005
                },
                "seed": 0,
                "input_dim": 1,
                "output_dim": 1
            }
            self.inverse_model = CustomNeuralNetwork(**inverse_model_params)
        if forward_model_params is None:
            forward_model_params = {
                "layers_info": {
                    "n_hidden_layers": 1,
                    "types": "linear",
                    "sizes": 1,
                    "hidden_activations": "relu",
                    "output_activation": "softmax",
                },
                "optimizer_info": {
                    "type": "adam",
                    "learning_rate": 0.005
                },
                "seed": 0,
                "input_dim": 1,
                "output_dim": 1
            }
            self.forward_model = CustomNeuralNetwork(**forward_model_params)

    def compute_icm_loss(self, batch, nn):
        self.check_models(nn)
        actor_actions_det = batch.actions.detach()
        # make states embeddings
        embedding_prev = self.embedder(batch.observations)
        embedding_next = self.embedder(batch.next_observations)

        # compute inverse model loss
        inverse_input = torch.cat((embedding_prev, embedding_next), dim=1)
        action_pred = self.inverse_model(inverse_input)
        inverse_loss = CrossEntropyLoss(action_pred, actor_actions_det[:, 0].to(torch.int64))

        # compute forward model loss
        forward_input = torch.cat((embedding_prev, actor_actions_det), dim=1)
        next_obs_pred = self.forward_model(forward_input)
        forward_loss = MSELoss(next_obs_pred, embedding_next)

        # compute the loss
        icm_loss = self.inv_coef * inverse_loss + (1 - self.inv_coef) * forward_loss
        
        # backporpagate the loss
        self.update_params(icm_loss)
    
    def get_intrinsic_reward(self, nn, obs, next_obs, action):
        self.check_models(nn)
        actor_action_det = torch.tensor([action])
        embedding_prev = self.embedder(obs)
        embedding_next = self.embedder(next_obs)

        inverse_input = torch.cat((embedding_prev, embedding_next))
        action_pred = self.inverse_model(inverse_input)
        inverse_loss = CrossEntropyLoss(action_pred.unsqueeze(0), actor_action_det)
        # compute forward model loss
        forward_input = torch.cat((embedding_prev, actor_action_det))
        next_obs_pred = self.forward_model(forward_input)
        forward_loss = MSELoss(next_obs_pred, embedding_next)
        # compute the loss
        icm_loss = self.inv_coef * inverse_loss + (1 - self.inv_coef) * forward_loss

        return icm_loss.detach()
        
    def check_models(self, nn):
        """function that check that the models have the good dimensions 
        by comparing them with the nn's dimensions. If the dimensions 
        don't match, the models are reinitialized with dimensions 
        mathing the nn's dimensions.

        Args:
            nn (CustomNeuralNetwork): the neural network that will be used
        """
        action_size = nn.output_dim
        observation_size = nn.input_dim
        if self.inverse_model.input_dim != self.embedding_size * 2 or \
            self.inverse_model.output_dim != action_size:
            self.inverse_model.reinit_layers(self.embedding_size * 2, action_size)
        if self.forward_model.input_dim != self.embedding_size + action_size or \
            self.forward_model.output_dim != self.embedding_size:
            self.forward_model.reinit_layers(
                self.embedding_size + action_size, self.embedding_size)
        if self.embedder == None:
            self.embedder = CustomNeuralNetwork(
                input_dim=observation_size,
                output_dim=self.embedding_size,
                layers_info={
                    "n_hidden_layers": 1,
                    "types": "linear",
                    "sizes": int(max(self.embedding_size, observation_size/2)),
                    "hidden_activations": "relu",
                    "output_activation": "none",
                },
                optimizer_info={
                    "type": "adam",
                    "learning_rate": 0.005
                }
            )

    def update_params(self, loss):
        """function that update the parameters of the models.

        Args:
            loss (torch.Tensor): the loss of the ICM
        """
        self.inverse_model.optimizer.zero_grad()
        self.forward_model.optimizer.zero_grad()
        self.embedder.optimizer.zero_grad()
        loss.backward()
        self.inverse_model.optimizer.step()
        self.forward_model.optimizer.step()
        self.embedder.optimizer.step()