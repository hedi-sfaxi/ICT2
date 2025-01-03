"""
Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated: April 24th, 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)
"""

# Necessary Packages
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple


def train_test_divide(data_x: List[np.ndarray], data_x_hat: List[np.ndarray], 
                      data_t: List[np.ndarray], data_t_hat: List[np.ndarray], 
                      train_rate: float = 0.8) -> Tuple[List[np.ndarray], List[np.ndarray], 
                                                         List[np.ndarray], List[np.ndarray], 
                                                         List[np.ndarray], List[np.ndarray], 
                                                         List[np.ndarray], List[np.ndarray]]:
    """
    Divide train and test data for both original and synthetic data.

    Args:
        data_x (List[np.ndarray]): Original data.
        data_x_hat (List[np.ndarray]): Generated data.
        data_t (List[np.ndarray]): Original time.
        data_t_hat (List[np.ndarray]): Generated time.
        train_rate (float): Ratio of training data (default is 0.8).

    Returns:
        Tuple: Contains the following datasets:
            - Train and test sets for original data (data_x, data_t).
            - Train and test sets for synthetic data (data_x_hat, data_t_hat).
    """
    # Split original data into train and test
    idx = np.random.permutation(len(data_x))
    train_idx = idx[:int(len(data_x) * train_rate)]
    test_idx = idx[int(len(data_x) * train_rate):]

    train_x = [data_x[i] for i in train_idx]
    test_x = [data_x[i] for i in test_idx]
    train_t = [data_t[i] for i in train_idx]
    test_t = [data_t[i] for i in test_idx]

    # Split synthetic data into train and test
    idx = np.random.permutation(len(data_x_hat))
    train_idx = idx[:int(len(data_x_hat) * train_rate)]
    test_idx = idx[int(len(data_x_hat) * train_rate):]

    train_x_hat = [data_x_hat[i] for i in train_idx]
    test_x_hat = [data_x_hat[i] for i in test_idx]
    train_t_hat = [data_t_hat[i] for i in train_idx]
    test_t_hat = [data_t_hat[i] for i in test_idx]

    return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat


def extract_time(data: List[np.ndarray]) -> Tuple[List[int], int]:
    """
    Extract maximum sequence length and sequence lengths for each sample in the data.

    Args:
        data (List[np.ndarray]): The original time-series data.

    Returns:
        Tuple:
            - List[int]: Sequence lengths for each data sample.
            - int: Maximum sequence length.
    """
    time = []
    max_seq_len = 0
    for sample in data:
        seq_len = len(sample[:, 0])  # Assuming 2D data where first column represents time.
        max_seq_len = max(max_seq_len, seq_len)
        time.append(seq_len)

    return time, max_seq_len


def rnn_cell(module_name: str, input_size: int, hidden_dim: int) -> nn.Module:
    """
    Create and return an RNN cell (GRU, LSTM, or LSTM with Layer Normalization).

    Args:
        module_name (str): The type of RNN cell ('gru', 'lstm', or 'lstmLN').
        hidden_dim (int): Number of hidden units in the RNN cell.

    Returns:
        nn.Module: The corresponding RNN cell.
    """
    assert module_name in ['gru', 'lstm', 'lstmLN']

    if module_name == 'gru':
        rnn_cell = nn.GRUCell(input_size=input_size, hidden_size=hidden_dim)
    elif module_name == 'lstm':
        rnn_cell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_dim)
    elif module_name == 'lstmLN':
        # PyTorch does not have LSTM with Layer Normalization by default,
        # but we can manually implement Layer Normalization within LSTM
        class LSTMCellLN(nn.Module):
            def __init__(self, input_size, hidden_size):
                super(LSTMCellLN, self).__init__()
                self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
                self.layer_norm = nn.LayerNorm(hidden_size)

            def forward(self, x, hx):
                h, c = self.lstm_cell(x, hx)
                h = self.layer_norm(h)
                return h, c

        rnn_cell = LSTMCellLN(input_size=input_size, hidden_size=hidden_dim)

    return rnn_cell


def random_generator(batch_size: int, z_dim: int, T_mb: List[int], max_seq_len: int) -> List[torch.Tensor]:
    """
    Generate random vectors for the GAN model.

    Args:
        batch_size (int): Size of the random vector batch.
        z_dim (int): Dimensionality of the random vector.
        T_mb (List[int]): Time information for the random vectors.
        max_seq_len (int): Maximum sequence length.

    Returns:
        List[torch.Tensor]: A list of generated random vectors.
    """
    Z_mb = []
    for i in range(batch_size):
        temp = torch.zeros(max_seq_len, z_dim)
        temp_Z = torch.rand(T_mb[i], z_dim)
        temp[:T_mb[i], :] = temp_Z
        Z_mb.append(temp_Z)
    
    return torch.tensor(np.array(Z_mb))


def batch_generator(data: List[np.ndarray], time: List[int], batch_size: int) -> Tuple[List[np.ndarray], List[int]]:
    """
    Generate mini-batches of time-series data and corresponding time information.

    Args:
        data (List[np.ndarray]): The time-series data.
        time (List[int]): Time information.
        batch_size (int): Number of samples per batch.

    Returns:
        Tuple:
            - List[np.ndarray]: Mini-batches of data.
            - List[int]: Corresponding time information for each batch.
    """
    idx = np.random.permutation(len(data))
    train_idx = idx[:batch_size]

    X_mb = torch.tensor(np.array([data[i] for i in train_idx]))
    T_mb = torch.tensor(np.array([time[i] for i in train_idx]))

    return X_mb, T_mb
