from pathlib import Path
from typing import List, Optional, Tuple, Union

import flwr as fl
from flwr.common import EvaluateIns, FitIns, FitRes, Parameters
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
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
from loguru import logger
import numpy as np
import pandas as pd

from src.sir_model import SirModel

NUM_ROW = 2000
LOOKAHEAD = 7


def to_parameters_ndarrays(numerator: float, denominator: float) -> List[np.ndarray]:
    """Utility function to convert SirModel parameters to List[np.ndarray] used by
    Flower's NumPyClient for transferring model parameters.
    """
    return [np.array([numerator, denominator])]


def from_parameters_ndarrays(parameters: List[np.ndarray]) -> Tuple[float, float]:
    """Utility function to convert SirModel parameters from List[np.ndarray] used by
    Flower's NumPyClient for transferring model parameters.
    """
    numerator, denominator = parameters[0]
    return numerator, denominator


def get_model_parameters(model: SirModel) -> List[np.ndarray]:
    """Gets the paramters of a SirModel model as List[np.ndarray] used by Flower's
    NumPyClient for transferring model parameters.
    """
    return to_parameters_ndarrays(model.numerator, model.denominator)


def set_model_parameters(model: SirModel, parameters: List[np.ndarray]) -> SirModel:
    """Sets the parameters of a SirModel model from a List[np.ndarray] used by Flower's
    NumPyClient for transferring model parameters."""
    numerator, denominator = from_parameters_ndarrays(parameters)
    model.set_params(numerator=numerator, denominator=denominator)
    return model


class TrainingClient(fl.client.NumPyClient):
    def __init__(self, cid: str, model: SirModel, disease_outcome_df: pd.DataFrame):
        super().__init__()
        self.cid = cid
        self.model = model
        self.disease_outcome_df = disease_outcome_df

    def fit(
        self, parameters: List[np.ndarray], config: dict
    ) -> Tuple[List[np.ndarray], int, dict]:
        """Fit model on partitioned dataset. Server is not passing any meaningful
        parameters or configuration. Returned fitted model parameters back to server."""
        self.model.fit(self.disease_outcome_df)
        return get_model_parameters(self.model), self.disease_outcome_df.shape[0], {}


def train_client_factory(
    cid: str,
    person_data_path: Path,
    household_data_path: Path,
    residence_location_data_path: Path,
    activity_location_data_path: Path,
    activity_location_assignment_data_path: Path,
    population_network_data_path: Path,
    disease_outcome_data_path: Path,
    client_dir: Path,
) -> Union[fl.client.Client, fl.client.NumPyClient]:
    """
    Factory function that instantiates and returns a Flower Client for training.
    The federated learning simulation engine will use this function to
    instantiate clients with all necessary dependencies.

    Args:
        cid (str): Identifier for a client node/federation unit. Will be
            constant over the simulation and between train and test stages.
        person_data_path (Path): Path to CSV data file for the Person table, for
            the partition specific to this client.
        household_data_path (Path): Path to CSV data file for the House table,
            for the partition specific to this client.
        residence_location_data_path (Path): Path to CSV data file for the
            Residence Locations table, for the partition specific to this
            client.
        activity_location_data_path (Path): Path to CSV data file for the
            Activity Locations on table, for the partition specific to this
            client.
        activity_location_assignment_data_path (Path): Path to CSV data file
            for the Activity Location Assignments table, for the partition
            specific to this client.
        population_network_data_path (Path): Path to CSV data file for the
            Population Network table, for the partition specific to this client.
        disease_outcome_data_path (Path): Path to CSV data file for the Disease
            Outcome table, for the partition specific to this client.
        client_dir (Path): Path to a directory specific to this client that is
            available over the simulation. Clients can use this directory for
            saving and reloading client state.

    Returns:
        (Union[Client, NumPyClient]): Instance of Flower Client or NumPyClient.
    """
    disease_outcome_df = pd.read_csv(disease_outcome_data_path).iloc[0:NUM_ROW]
    model = SirModel(lookahead=LOOKAHEAD)
    return TrainingClient(cid=cid, model=model, disease_outcome_df=disease_outcome_df)


class TrainStrategy(fl.server.strategy.Strategy):
    """Federated aggregation equivalent to pooling observations across partitions."""

    def __init__(self, server_dir: Path):
        self.server_dir = server_dir
        super().__init__()

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

    def aggregate(self, results):
        """Compute weighted average."""
        # Define (TMP) Laplace Matrix
        L_support = self.Laplacian_Matrix(len(results),topology='Ring')
        # I am not sure the weights update law here is right or not. My update law would be X = np.matmul(L,X).
        # With our update law, After several steps, the value of X would converge to a single value.

        # Create a list of weights, each multiplied by the related number of examples
        weighted_weights = []
        for weights in results: # (numerator, denom)
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
            weights_prime = []
            for layer_updates in zip(*weighted_weights): # iterate numer/denom
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
        numerator, denominator = 0, 0
        for param in params:
            numerator = numerator + param[0]
            denominator = denominator + param[1]
        numerator = numerator / len(params)
        denominator = denominator / len(params)
        return [numerator, denominator]

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

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters:
        """Do nothing. Return empty Flower Parameters dataclass."""
        return fl.common.ndarrays_to_parameters([])

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Fit all clients."""
        logger.info(f"Configuring fit for round {server_round}...")
        # Fit every client once. Don't need to pass any initial parameters or config.
        clients = list(client_manager.all().values())
        empty_fit_ins = fl.common.FitIns(fl.common.ndarrays_to_parameters([]), {})
        logger.info(f"...done configuring fit for round {server_round}")
        
        self.parameters_for_client = []
        self.ids_for_client = []
        return [(client, empty_fit_ins) for client in clients]

    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures
    ) -> Tuple[Optional[Parameters], dict]:
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        # if failures:
        #     return None, {}

        """Aggregate fit results by summing the numerator and denominator of beta
        estimates."""
        if len(failures) > 0:
            raise Exception(f"Client fit round had {len(failures)} failures.")

        list_client_id = []
        for i in range(len(results)):
            list_client_id.append(results[i][0])
        self.ids_for_client = list_client_id

        # results is List[Tuple[ClientProxy, FitRes]]
        # convert FitRes to List[np.ndarray]
        client_parameters_list_of_ndarrays = [
            fl.common.parameters_to_ndarrays(fit_res.parameters)
            for _, fit_res in results
        ]
        # get numerators and denominators out of List[np.ndarray]
        client_numerators, client_denominators = zip(
            *[
                from_parameters_ndarrays(params_list_of_ndarrays)
                for params_list_of_ndarrays in client_parameters_list_of_ndarrays
            ]
        )

        client_weights = list(zip(client_numerators, client_denominators))

        # aggregate by summing running numerator and denominator sums
        # numerator = sum(client_numerators)
        # denominator = sum(client_denominators)

        # split weight (of clients) for each server
        weights_results_foreach_subserver = self.split_weight_result(client_weights, self.maximum_servers)

        # aggregate parameters
        for i in range(len(weights_results_foreach_subserver)):
            # i is the i-th sub server
            # self.aggregate performs distributed learning and return weight of random client in the subserver
            # deleted ndarrays_to_parameters, process this later (numerator and denominator)
            parameters_aggregated_centroid = self.aggregate(weights_results_foreach_subserver[i])
            self.parameters_for_server.append(parameters_aggregated_centroid)

        # Post process parameter list
        # if len(self.parameters_for_client) != 0:
        #     for i in range(len(self.parameters_for_client)):
        #         self.parameters_for_client[i] = ndarrays_to_parameters(self.parameters_for_client[i])

        # convert back to List[np.ndarray] then Parameters dataclass to send to clients
        # parameters = fl.common.ndarrays_to_parameters(
        #     to_parameters_ndarrays(numerator=numerator, denominator=denominator)
        # )
        parameters_aggregated_all_servers = self.aggregate_among_server(self.parameters_for_server)
        parameters = fl.common.ndarrays_to_parameters(
            to_parameters_ndarrays(numerator=parameters_aggregated_all_servers[0], denominator=parameters_aggregated_all_servers[1])
        )
        self.parameters_for_server = []
        return parameters, {}

    def configure_evaluate(self, server_round: int, parameters, client_manager):
        """Do nothing. Return empty list."""
        return []

    def aggregate_evaluate(self, server_round: int, results, failures):
        """Do nothing. Expect no results to aggregate. Return empty results."""
        return None, {}

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, dict]]:
        """Write model to disk. No actual evaluation."""
        # server_round=0 is evaluates an initial centralized model.
        # We don't initialize one, so we do nothing.
        if server_round > 0:
            numerator, denominator = from_parameters_ndarrays(
                fl.common.parameters_to_ndarrays(parameters)
            )
            model = SirModel(numerator=numerator, denominator=denominator)
            checkpoint_name = f"model-{server_round:02}.json"
            model.save(self.server_dir / checkpoint_name)
            logger.info(f"Model checkpoint {checkpoint_name} saved to disk by server.")
        return None


def train_strategy_factory(
    server_dir: Path,
) -> Tuple[fl.server.strategy.Strategy, int]:
    """
    Factory function that instantiates and returns a Flower Strategy, plus the
    number of federated learning rounds to run.

    Args:
        server_dir (Path): Path to a directory specific to the server/aggregator
            that is available over the simulation. The server can use this
            directory for saving and reloading server state. Using this
            directory is required for the trained model to be persisted between
            training and test stages.

    Returns:
        (Strategy): Instance of Flower Strategy.
        (int): Number of federated learning rounds to execute.
    """
    training_strategy = TrainStrategy(server_dir=server_dir)
    num_rounds = 1
    return training_strategy, num_rounds


class TestClient(fl.client.NumPyClient):
    """Custom Flower NumPyClient class for test."""

    def __init__(
        self,
        cid: str,
        model: SirModel,
        disease_outcome_df: pd.DataFrame,
        preds_format_df: pd.DataFrame,
        preds_path: Path,
    ):
        super().__init__()
        self.cid = cid
        self.model = model
        self.disease_outcome_df = disease_outcome_df
        self.preds_format_df = preds_format_df
        self.preds_path = preds_path

    def evaluate(self, parameters: Parameters, config: dict) -> Tuple[float, int, dict]:
        """Make predictions on the test split. Use model parameters from server."""
        set_model_parameters(self.model, parameters)
        predictions = self.model.predict(self.disease_outcome_df)
        predictions.loc[self.preds_format_df.index].to_csv(self.preds_path)
        logger.info(f"Client test predictions saved to disk for client {self.cid}.")
        # Return empty metrics. We're not actually evaluating anything
        return 0.0, 0, {}


def test_client_factory(
    cid: str,
    person_data_path: Path,
    household_data_path: Path,
    residence_location_data_path: Path,
    activity_location_data_path: Path,
    activity_location_assignment_data_path: Path,
    population_network_data_path: Path,
    disease_outcome_data_path: Path,
    client_dir: Path,
    preds_format_path: Path,
    preds_dest_path: Path,
) -> Union[fl.client.Client, fl.client.NumPyClient]:
    """
    Factory function that instantiates and returns a Flower Client for test-time
    inference. The federated learning simulation engine will use this function
    to instantiate clients with all necessary dependencies.

    Args:
        cid (str): Identifier for a client node/federation unit. Will be
            constant over the simulation and between train and test stages.
        person_data_path (Path): Path to CSV data file for the Person table, for
            the partition specific to this client.
        household_data_path (Path): Path to CSV data file for the House table,
            for the partition specific to this client.
        residence_location_data_path (Path): Path to CSV data file for the
            Residence Locations table, for the partition specific to this
            client.
        activity_location_data_path (Path): Path to CSV data file for the
            Activity Locations on table, for the partition specific to this
            client.
        activity_location_assignment_data_path (Path): Path to CSV data file
            for the Activity Location Assignments table, for the partition
            specific to this client.
        population_network_data_path (Path): Path to CSV data file for the
            Population Network table, for the partition specific to this client.
        disease_outcome_data_path (Path): Path to CSV data file for the Disease
            Outcome table, for the partition specific to this client.
        client_dir (Path): Path to a directory specific to this client that is
            available over the simulation. Clients can use this directory for
            saving and reloading client state.
        preds_format_path (Path): Path to CSV file matching the format you must
            write your predictions with, filled with dummy values.
        preds_dest_path (Path): Destination path that you must write your test
            predictions to as a CSV file.

    Returns:
        (Union[Client, NumPyClient]): Instance of Flower Client or NumPyClient.
    """
    model = SirModel()
    disease_outcome_df = pd.read_csv(disease_outcome_data_path).iloc[0:NUM_ROW]
    preds_format_df = pd.read_csv(preds_format_path, index_col="pid")
    return TestClient(
        cid=cid,
        model=model,
        disease_outcome_df=disease_outcome_df,
        preds_format_df=preds_format_df,
        preds_path=preds_dest_path,
    )


class TestStrategy(fl.server.strategy.Strategy):
    """Custom Flower strategy for test."""

    def __init__(self, server_dir: Path):
        self.server_dir = server_dir
        super().__init__()

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters:
        """Load saved model parameters from training."""
        logger.info("Loading saved model from checkpoint...")
        last_checkpoint_path = sorted(self.server_dir.glob("model-*.json"))[-1]
        model = SirModel.load(last_checkpoint_path)
        parameters_ndarrays = to_parameters_ndarrays(model.numerator, model.denominator)
        parameters = fl.common.ndarrays_to_parameters(parameters_ndarrays)
        logger.info(
            f"Model parameters loaded from checkpoint {last_checkpoint_path.name}"
        )
        return parameters

    def configure_fit(self, server_round, parameters, client_manager):
        """Do nothing and return empty list. We don't need to fit clients for test."""
        return []

    def aggregate_fit(self, server_round, results, failures):
        """Do nothing and return empty results. No fit results to aggregate for test."""
        return None, {}

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Run evaluate on all clients to make test predictions."""
        evaluate_ins = fl.common.EvaluateIns(parameters, {})
        clients = list(client_manager.all().values())
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(self, server_round, results, failures):
        """Do nothing and return empty results. Not actually evaluating any metrics."""
        return None, {}

    def evaluate(self, server_round, parameters):
        return None


def test_strategy_factory(
    server_dir: Path,
) -> Tuple[fl.server.strategy.Strategy, int]:
    """
    Factory function that instantiates and returns a Flower Strategy, plus the
    number of federation rounds to run.

    Args:
        server_dir (Path): Path to a directory specific to the server/aggregator
            that is available over the simulation. The server can use this
            directory for saving and reloading server state. Using this
            directory is required for the trained model to be persisted between
            training and test stages.

    Returns:
        (Strategy): Instance of Flower Strategy.
        (int): Number of federated learning rounds to execute.
    """
    test_strategy = TestStrategy(server_dir=server_dir)
    num_rounds = 1
    return test_strategy, num_rounds
