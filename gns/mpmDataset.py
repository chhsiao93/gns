from typing import Tuple, Union
import json
import numpy as np
import torch

from FunctionEncoder.Dataset.BaseDataset import BaseDataset


class SandDataset(BaseDataset):

    def __init__(self,
                 path,
                 n_input_sequence = 5,
                 dim = 2,
                 n_functions_per_sample:int = 10,
                 n_examples_per_sample:int = 1_000,
                 n_points_per_sample:int = 10_000,
                 device:str="auto",
                 dtype:torch.dtype = torch.float32,
                 ):
        super().__init__(input_size=((n_input_sequence)*dim + dim**2 + 1,), # (vel_hist + dist_to_boundary + friction_angle)
                         output_size=(2,),
                         total_n_functions=float('inf'),
                         total_n_samples_per_function=float('inf'),
                         data_type="deterministic",
                         n_functions_per_sample=n_functions_per_sample,
                         n_examples_per_sample=n_examples_per_sample,
                         n_points_per_sample=n_points_per_sample,
                         device=device,
                         dtype=dtype,
                         )
        self.n_input_sequence = n_input_sequence
        self.dim = dim
        self.data = np.load(path, allow_pickle=True)
        
                                                                                                                              
        
    def preprocess(self, trajectory, idx):
        n_frames, n_particles, dim = trajectory['positions'].shape
        assert idx >= self.n_input_sequence and idx < n_frames
        input_sequence = trajectory['positions'][idx-self.n_input_sequence:idx+2] # (n_input_sequence+2, n_particles, dim)
        nparticles = input_sequence.shape[1]
        # current position
        current_position = input_sequence[-2]
        # next position
        next_position = input_sequence[-1]
        # compute distance to boundary
        boundary = np.array([[0.1, 0.9], [0.1, 0.9]])
        dist_to_lower_boudary = current_position - boundary[:, 0]
        dist_to_upper_boudary = boundary[:, 1] - current_position
        dist_to_boundary = np.concatenate([dist_to_lower_boudary, dist_to_upper_boudary], axis=-1)
        assert dist_to_boundary.shape == (n_particles, dim * 2) # should be (n_particles, 2*dim)
        # friction angle
        friction_angle = np.full((nparticles,1), trajectory["friction_angle"]) # (n_particles,1)
        
        # compute velocity
        velocity_sequence = np.zeros_like(input_sequence) # (n_input_sequence+2, n_particles, dim)
        velocity_sequence[1:] = input_sequence[1:] - input_sequence[:-1]
        # compute next acceleration
        next_acceleration = velocity_sequence[-1] - velocity_sequence[-2] # (n_particles, dim)
        # reshape velocity_sequence to (n_particles, n_input_sequence*dim)
        velocity_sequence = np.swapaxes(velocity_sequence[1:-1], 0, 1) # (n_particles, n_input_sequence, dim)
        velocity_sequence = velocity_sequence.reshape(nparticles, -1) # (n_particles, n_input_sequence*dim)
        input = np.concatenate((velocity_sequence,  dist_to_boundary, friction_angle), axis=-1) # (n_particles ,n_input_sequence*dim + 2*dim + 1)
        output = next_acceleration # (n_particles, dim)
        return input, output
    
        

    def sample(self) -> Tuple[  torch.tensor, 
                                torch.tensor, 
                                torch.tensor, 
                                torch.tensor, 
                                dict]:
        with torch.no_grad():
            n_functions = self.n_functions_per_sample
            n_examples = self.n_examples_per_sample
            n_points = self.n_points_per_sample

            # sample n sequence of inputs
            idxs = np.random.randint(self.n_input_sequence+1, len(self.data["positions"])-2, n_functions)
            inputs, outputs = [], []
            example_inputs, example_outputs = [], []
            for idx in idxs:
                input, output = self.preprocess(self.data, idx)
                example_input, example_output = self.preprocess(self.data, idx-1)
                inputs.append(input)
                outputs.append(output)
                example_inputs.append(example_input)
                example_outputs.append(example_output)
            inputs = np.stack(inputs)
            outputs = np.stack(outputs)
            example_inputs = np.stack(example_inputs)
            example_outputs = np.stack(example_outputs)
            assert inputs.shape == (n_functions, n_points, *self.input_size)
            assert outputs.shape == (n_functions, n_points, self.dim)
            assert example_inputs.shape == (n_functions, n_examples, *self.input_size)
            assert example_outputs.shape == (n_functions, n_examples, self.dim)
            xs = torch.tensor(inputs, dtype=self.dtype, device=self.device)
            ys = torch.tensor(outputs, dtype=self.dtype, device=self.device)
            example_xs = torch.tensor(example_inputs, dtype=self.dtype, device=self.device)
            example_ys = torch.tensor(example_outputs, dtype=self.dtype, device=self.device)
            

            return example_xs, example_ys, xs, ys, {'idxs': idxs}
    
    def get_one_frame(self, idx):
        input, output = self.preprocess(self.data, idx)
        input = torch.tensor(input, dtype=self.dtype, device=self.device).unsqueeze(0)
        output = torch.tensor(output, dtype=self.dtype, device=self.device).unsqueeze(0)
        return input, output




class SandPairDataset(BaseDataset):

    def __init__(self,
                 path,
                 n_input_sequence = 6,
                 dim = 2,
                 n_functions:int = 10,
                 n_examples:int = 50,
                 n_queries:int = 150,
                 device:str="auto",
                 dtype:torch.dtype = torch.float32,
                 ):
        super().__init__(input_size=(2*((n_input_sequence-1)*dim + dim**2 + 1)+3,), # 2*(vel_hist + dist_to_boundary + material_type) + (pair_rel_position + pair_rel_distance)
                         output_size=(2,),
                         data_type="deterministic",
                         n_functions=n_functions,
                         n_examples=n_examples,
                         n_queries=n_queries,
                         device=device,
                         dtype=dtype,
                         )
        self.dt = 1.0
        self.n_input_sequence = n_input_sequence
        self.dim = dim
        
        npz_path = path + "/examples.npz"
        metadata_path = path + "/metadata.json"
        
        self.data = self.load_npz(npz_path)
        self.n_trajectories = len(self.data)
        self.metadata = self.load_metadata(metadata_path)
        
        
    def load_npz(self, path):
        # loading data
        with np.load(path, allow_pickle=True) as data_file:
            if 'gns_data' in data_file:
                data = data_file['gns_data']
            else:
                data = [item for _, item in data_file.items()]
        return data
    
    def load_metadata(self, path):
        # loading metadata
        with open(path, 'r') as f:
            metadata = json.load(f)
        self.bounds = metadata["fe_train"]['bounds']
        self.trajectory_lengths = metadata["fe_train"]['sequence_length']
        self.connectivity_radius = metadata["train"]['default_connectivity_radius']
        # Normalization stats
        self.normalization_stats = {
            'acceleration': {
                'mean': np.array(metadata["fe_train"]['acc_mean']),
                'std': np.array(metadata["fe_train"]['acc_std']),
            },
            'velocity': {
                'mean': np.array(metadata["fe_train"]['vel_mean']),
                'std': np.array(metadata["fe_train"]['vel_std']),
            },
        }
        return metadata
                                                                                                        
    def preprocess(self,
                   position_squence,
                   particle_types,
                   next_position=None):
        
        n_input_sequence = position_squence.shape[0]
        n_particles = position_squence.shape[1]
        dim = position_squence.shape[-1]
        assert n_particles == 2
        assert n_input_sequence == self.n_input_sequence
        current_position = position_squence[-1,:]
        
        # compute distance to boundary
        boundary = np.array(self.bounds)
        dist_to_lower_boudary = current_position - boundary[:, 0]
        dist_to_upper_boudary = boundary[:, 1] - current_position
        dist_to_boundary = np.concatenate([dist_to_lower_boudary, dist_to_upper_boudary], axis=-1)
        assert dist_to_boundary.shape == (n_particles, dim * 2) # should be (n_particles, 2*dim)
        
        # compute velocity
        velocity_sequence= position_squence[1:] - position_squence[:-1]
        if next_position is not None:
            # compute next acceleration
            next_velocity = next_position - current_position
            next_acceleration = next_velocity - velocity_sequence[-1] # (n_particles, dim)
            # normalize next_acceleration
            acc_stats = self.normalization_stats['acceleration']
            next_acceleration = (next_acceleration - acc_stats['mean']) / acc_stats['std']
            output = next_acceleration[0]
        
        # normalize velocity
        velocity_stats = self.normalization_stats['velocity']
        velocity_sequence = (velocity_sequence - velocity_stats['mean']) / velocity_stats['std']
        # reshape velocity_sequence to (n_particles, (n_input_sequence-1)*dim)
        velocity_sequence = np.swapaxes(velocity_sequence, 0, 1) # (n_particles, (n_input_sequence-1), dim)
        velocity_sequence = velocity_sequence.reshape(n_particles, -1) # (n_particles, (n_input_sequence-1)*dim)
        
        # compute distance between the pair
        pair_rel_position = current_position[1] - current_position[0]
        pair_rel_distance = np.linalg.norm(pair_rel_position)
        # normalize distance
        pair_rel_position = pair_rel_position / self.connectivity_radius
        pair_rel_distance = pair_rel_distance / self.connectivity_radius
        
        node_feature_1 = np.concatenate((velocity_sequence[0],  dist_to_boundary[0], np.array([particle_types[0]])), axis=-1)
        node_feature_2 = np.concatenate((velocity_sequence[1],  dist_to_boundary[1], np.array([particle_types[1]])), axis=-1)
        edge_feature = np.concatenate((pair_rel_position, np.array([pair_rel_distance])), axis=-1)
        input = np.concatenate((node_feature_1, node_feature_2,  edge_feature), axis=-1)
        if next_position is None:
            # if there is no next position, meaning we only want input features
            return input
        else:
            # if there is next position, meaning we want input features and target acceleration
            return input, output

    def get_data_from_trajectory(self, trajectory):
        # get the first trajectory
        position_trajectory = trajectory[0]
        particle_types = trajectory[1]
        inputs, outputs = [], []
        for idx in range(self.n_input_sequence, self.trajectory_lengths):
            position_squence = position_trajectory[idx-self.n_input_sequence:idx]
            next_position = position_trajectory[idx]
            # checking dimension
            assert position_squence.shape == (self.n_input_sequence, 2, self.dim) # position_squence should be (n_input_sequence, n_particles, dim)
            assert next_position.shape == (2, self.dim) # next_position should be (n_particles, dim)
            assert particle_types.shape == (2,), f"Expected shape to be (2,), but got {particle_types.shape}" # particle_types should be (n_particles)
            input, output = self.preprocess(position_squence, particle_types, next_position)
            inputs.append(input)
            outputs.append(output)
            # exchange the information of particle 1 and 2
            position_squence = np.flip(position_squence, axis=1)
            next_position = np.flip(next_position, axis=0)
            particle_types = np.flip(particle_types, axis=0)
            input, output = self.preprocess(position_squence, particle_types, next_position)
            inputs.append(input)
            outputs.append(output)
            
        return np.stack(inputs), np.stack(outputs)
    
    def sample(self) -> Tuple[  torch.tensor, 
                                torch.tensor, 
                                torch.tensor, 
                                torch.tensor, 
                                dict]:
        with torch.no_grad():
            n_functions = self.n_functions_per_sample
            n_examples = self.n_examples_per_sample
            n_points = self.n_points_per_sample

            # sample n trajectories
            function_idxs = np.random.choice(self.n_trajectories, n_functions)
            assert n_examples + n_points <= 2*(self.trajectory_lengths-self.n_input_sequence-1)
            # print(n_functions, n_examples, n_points)
            
            inputs, outputs = [], []
            example_inputs, example_outputs = [], []
            for idx in function_idxs:
                input, output = self.get_data_from_trajectory(self.data[idx])
                data_idxs = np.random.choice(2*(self.trajectory_lengths-self.n_input_sequence-1), n_examples+n_points)
                example_idxs = data_idxs[:n_examples]
                point_idxs = data_idxs[n_examples:]
                inputs.append(input[point_idxs])
                outputs.append(output[point_idxs])
                example_inputs.append(input[example_idxs])
                example_outputs.append(output[example_idxs])
            # print(len(data_idxs),len(example_idxs), len(point_idxs))
            inputs = np.stack(inputs)
            outputs = np.stack(outputs)
            example_inputs = np.stack(example_inputs)
            example_outputs = np.stack(example_outputs)
            assert inputs.shape == (n_functions, n_points, *self.input_size) , f"inputs shape {inputs.shape} is not equal to ({n_functions, n_points, *self.input_size})"
            assert outputs.shape == (n_functions, n_points, self.dim)
            assert example_inputs.shape == (n_functions, n_examples, *self.input_size)
            assert example_outputs.shape == (n_functions, n_examples, self.dim)
            xs = torch.tensor(inputs, dtype=self.dtype, device=self.device)
            ys = torch.tensor(outputs, dtype=self.dtype, device=self.device)
            example_xs = torch.tensor(example_inputs, dtype=self.dtype, device=self.device)
            example_ys = torch.tensor(example_outputs, dtype=self.dtype, device=self.device)
            

            return example_xs, example_ys, xs, ys, {'idxs': function_idxs}

    def training_rollout(self, example_trajectories, target_trajectories):
        pass