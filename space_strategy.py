from collections import OrderedDict
from typing import List, Tuple

import flwr as fl
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from flwr.common import Metrics
from torch.utils.data import DataLoader, random_split

from logging import WARNING
from flwr.server.strategy import Strategy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from typing import Callable, Dict, List, Optional, Tuple, Union

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from functools import reduce

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def weighted_average_fit_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    loss_all = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"loss": sum(loss_all) / sum(examples)}


class strategy_custom(fl.server.strategy.FedAvg):
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        # Add additional arguments below
        self.parameters_for_client = []
        self.ids_for_client = []
        self.parameters_for_server = [] # to N and update
        self.maximum_servers = 2

    def Laplacian_Matrix(self, NUM_AGENTS, topology='Ring'):
        if topology == 'No':
            L = np.eye(NUM_AGENTS)

        if topology == 'Ring':
            L = 0.5 * np.eye(NUM_AGENTS) + 0.25 * np.eye(NUM_AGENTS, k=1) + 0.25 * np.eye(NUM_AGENTS,
                                                                                          k=-1) + 0.25 * np.eye(
                NUM_AGENTS, k=NUM_AGENTS - 1) + 0.25 * np.eye(NUM_AGENTS, k=-NUM_AGENTS + 1)

        if topology == 'Full':
            A = np.ones([NUM_AGENTS, NUM_AGENTS]) - np.eye(NUM_AGENTS)
            L = (A + sum(A[0]) * np.eye(NUM_AGENTS)) / sum(A + sum(A[0]) * np.eye(NUM_AGENTS))

        if topology == 'MS':
            A = np.random.randint(2, size=NUM_AGENTS * NUM_AGENTS)
            A = (np.ones([NUM_AGENTS, NUM_AGENTS]) - np.eye(NUM_AGENTS)) * A.reshape([NUM_AGENTS, NUM_AGENTS])
            vec = A + np.diag(A.sum(axis=1))
            zero_id = np.where(vec.sum(axis=1) == 0)
            for k in range(len(zero_id[0])):
                vec[zero_id[0][k]][zero_id[0][k]] = 1
            L = vec / vec.sum(axis=1).reshape(-1, 1)
        return L
        
    def aggregate(self, results: List[Tuple[NDArrays, int]]) -> NDArrays:
        """Compute weighted average."""
        # Calculate the total number of examples used during training
        num_examples_total = sum([num_examples for _, num_examples in results])

        # Define (TMP) Laplace Matrix
        L_support = self.Laplacian_Matrix(len(results),topology='Ring')
        # I am not sure the weights update law here is right or not. My update law would be X = np.matmul(L,X).
        # With our update law, After several steps, the value of X would converge to a single value.


        # L_support = np.zeros_like(L_support)

        # Create a list of weights, each multiplied by the related number of examples
        # weighted_weights = [
        #     [layer * num_examples for layer in weights] for weights, num_examples in results
        # ]
        weighted_weights = []
        for weights, num_examples in results:
            support = []
            for layer in weights:
                # support.append(layer * num_examples)
                support.append(layer)
            weighted_weights.append(support)

        # Compute average weights of each layer
        # weights_prime: NDArrays = [
        #     reduce(np.add, layer_updates) / num_examples_total
        #     for layer_updates in zip(*weighted_weights)
        # ]
        weights_prime_list = []
        for message_id in range(len(results)):
            weights_prime: NDArrays = []
            for layer_updates in zip(*weighted_weights):
                # print(len(layer_updates)) # number of clients
                accumulate_sum = np.zeros_like(layer_updates[0])
                for i in range(len(layer_updates)):
                    accumulate_sum = np.add(accumulate_sum, L_support[message_id, i] * layer_updates[i])
                # support = reduce(np.add, layer_updates) / num_examples_total
                # Do we need to divide by num_examples_total if in dl scheme?
                # support = accumulate_sum / num_examples_total
                support = accumulate_sum
                weights_prime.append(support)
            weights_prime_list.append(weights_prime)

        for i in range(len(weights_prime_list)):
            self.parameters_for_client.append(weights_prime_list[i])

        # Temporarily choose 0 for global model update, need discuss this later
        return weights_prime_list[0]

    def aggregate_among_server(self, params):
        output = np.zeros_like(parameters_to_ndarrays(params[0]))
        for param in params:
            output = np.add(output, parameters_to_ndarrays(param))
        return ndarrays_to_parameters(output)

    def split_weight_result(self, params, server_num):
        output = []
        client_num = len(params)
        if server_num > client_num:
            print('server number greater than client number error!')
            server_num = client_num
        pre_calc = []
        for i in range(server_num):
            pre_calc.append(0)
        index = 0
        while index < client_num:
            pre_calc[index % server_num] += 1
            index += 1
        start = 0
        for i in range(server_num):
            end = start + pre_calc[i]
            output.append(params[start:end])
            start = end
        return output
        
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        # print(results[0][0], results[1][0]) # <flwr.server.grpc_server.grpc_client_proxy.GrpcClientProxy object at 0x000002548A0FFB48>
        list_client_id = []
        for i in range(len(results)):
            list_client_id.append(results[i][0])
        self.ids_for_client = list_client_id
        # print(len(weights_results)) # 2: two clients have weight
        # print(len(weights_results[0])) # 2: 0 indicates parameters, 1 indicates num_examples
        # print(len(weights_results[0][0])) # 16: for each parameter set, 16 layers have parameters
        # print(weights_results[0][0][0].shape)
        weights_results_foreach_subserver = self.split_weight_result(weights_results, self.maximum_servers)
        for i in range(len(weights_results_foreach_subserver)):
            parameters_aggregated_centroid = ndarrays_to_parameters(self.aggregate(weights_results_foreach_subserver[i])) # picked summed parameters
            self.parameters_for_server.append(parameters_aggregated_centroid)

        # Post process parameter list
        if len(self.parameters_for_client) != 0:
            for i in range(len(self.parameters_for_client)):
                self.parameters_for_client[i] = ndarrays_to_parameters(self.parameters_for_client[i])

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        # save all clients' parameter for one round, acc N round to update
        # self.parameters_for_server.append(self.parameters_for_client)
        if len(self.parameters_for_server) == self.maximum_servers:
            print('Aggregating all server parameters and update default.')
            parameters_aggregated_all_servers = self.aggregate_among_server(self.parameters_for_server)
            self.parameters_for_server = []
            return parameters_aggregated_all_servers, metrics_aggregated
        else:
            print('Error')
            return None, None

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)
        fit_ins_default = FitIns(parameters, config)

        # Post process FitIns for each client
        fit_ins_foreach_client = []
        # print(len(self.parameters_for_client))
        if len(self.parameters_for_client) != 0:
            for i in range(len(self.parameters_for_client)):
                fit_ins_foreach_client.append(FitIns(self.parameters_for_client[i], config))
        else:
            # default global parameter configuration
            fit_ins_foreach_client.append(fit_ins)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        # clients = client_manager.sample(
        #     num_clients=sample_size, min_num_clients=min_num_clients
        # )
        clients = client_manager.sample(
            num_clients=min_num_clients, min_num_clients=2 # set min sample client and max sample client
        )

        # Return client/config pairs
        client_pairs = []
    
        client_table = [0 for _ in clients]
        for idx_enumerator, client in enumerate(clients):
            flag = 0
            for client_idx in range(len(self.ids_for_client)):
                # if found the corresponding client in the parameter list
                if client == self.ids_for_client[client_idx]:
                    client_pairs.append((client, fit_ins_foreach_client[client_idx]))
                    client_table[idx_enumerator] = 1
                    flag = 1
                    # print('found:{}'.format(client))
                    break
            if flag == 0:
                client_pairs.append((client, fit_ins_default))
            
        # if len(self.ids_for_client) != 0:
        #     print(self.ids_for_client[0] == clients[0])
        #     print(self.ids_for_client[0] == clients[1])
        # print(self.ids_for_client)
        self.parameters_for_client = []
        self.ids_for_client = []
        return client_pairs

if __name__ == "__main__":
    strategy = strategy_custom(
            fraction_fit=1.0,
            fraction_evaluate=0.5,
            min_fit_clients=6,
            min_evaluate_clients=6,
            min_available_clients=6,
            evaluate_metrics_aggregation_fn = weighted_average,
            # fit_metrics_aggregation_fn = weighted_average_fit_metrics_aggregation_fn,
    )

    fl.server.start_server(server_address="127.0.0.1:8080", config=fl.server.ServerConfig(num_rounds=6), strategy=strategy)