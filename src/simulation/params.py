from typing import Optional
from src.simulation.graphs import adjacency_mat, friendliness_mat
from src.simulation.initsim import init_prior_distrs
from src.simulation.sim import run_simulation, SimParams, AIParams, DEFAULT_MAX_STEPS, LikelihoodGenerator, MaskedAILikelihoodGenerator, DEFAULT_DISCRETISED_BIAS
from graph_tool import Graph
from functools import cached_property
from dataclasses import dataclass, field, KW_ONLY
import numpy as np

CoinsList = list[int]

@dataclass
class CoinParams:
    bias: float = 0.6
    max_steps: int = DEFAULT_MAX_STEPS

    def get(self) -> CoinsList:
        ...

    def to_generated(self):
        return GeneratedCoinParams(
            bias=self.bias,
            max_steps=self.max_steps,
            coins=self.get(),
            generated_from=self,
        )
    
@dataclass
class RandomCoinParams(CoinParams):
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def get(self) -> CoinsList:
        # return (self.rng.random(self.max_steps) < self.bias).astype(int).tolist()
        return [1 if self.rng.random() < self.bias else 0 for _ in range(self.max_steps)]

@dataclass
class GeneratedCoinParams(CoinParams):
    _: KW_ONLY
    coins: CoinsList
    generated_from: Optional[CoinParams] = None
    
    def get(self) -> CoinsList:
        return self.coins
    
    @classmethod
    def from_coin_params(cls, coin_params: CoinParams):
        return cls(
            coins = coin_params.get(),
            bias = coin_params.bias,
            max_steps = coin_params.max_steps,
            generated_from=coin_params
        )

@dataclass
class PriorParams:
    n: int
    partisans: list[int] = field(default_factory=list)
    # TODO: cleanup, remove these from base class
    partisan_mean: float | list[float] = 0.3
    partisan_sd: float | list[float] = 0.01
    
    # TODO: should be abstractclass?
    def get(self) -> tuple[np.array, np.array]:
        ...
    
    @property
    def persuadable_agents(self):
        return [i for i in range(self.n) if i not in self.partisans]

    @property
    def is_duelling_partisans(self):
        return isinstance(self.partisan_mean, list) and isinstance(self.partisan_sd, list) \
            and len(self.partisan_mean) == len(self.partisans) \
            and len(self.partisan_sd) == len(self.partisans)

    def _apply_partisans(self, init_mean, init_sd):
        if isinstance(self.partisan_mean, list):
            assert len(self.partisan_mean) == len(self.partisans), f"partisan_mean len mismatch {len(self.partisan_mean)} != {len(self.partisans)}"
            assert len(self.partisan_sd) == len(self.partisans), f"partisan_sd len mismatch {len(self.partisan_sd)} != {len(self.partisans)}"

            for p, mean, sd in zip(self.partisans, self.partisan_mean, self.partisan_sd):
                init_mean[p] = mean
                init_sd[p] = sd
        
        elif isinstance(self.partisan_mean, float):
            assert isinstance(self.partisan_sd, float), "partisan_sd must be float if partisan_mean is float"
            
            init_mean[self.partisans] = self.partisan_mean
            init_sd[self.partisans] = self.partisan_sd

            # equivalent to, and tested against:
            # for p in self.partisans:
                # init_mean[p] = self.partisan_mean
                # init_sd[p] = self.partisan_sd

        else:
            raise ValueError("Invalid type for partisan_mean")
        
        return init_mean, init_sd

    def copy(self):
        return self.__class__(**self.__dict__)

@dataclass
class UniformRandomPriorParams(PriorParams):
    init_mean_range: tuple[float, float] = (0, 1)
    init_sd_range: tuple[float, float] = (0.2, 0.8)
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def get(self):
        init_mean = self.rng.uniform(*self.init_mean_range, self.n)
        init_sd = self.rng.uniform(*self.init_sd_range, self.n)
        
        return self._apply_partisans(init_mean, init_sd)

@dataclass
class GeneratedPriorParams(PriorParams):
    _: KW_ONLY
    init_mean: np.array
    init_sd: np.array
    generated_from: Optional[PriorParams] = None

    def get(self):
        return self.init_mean, self.init_sd
    
    def apply_partisans(self):
        self.init_mean, self.init_sd = PriorParams._apply_partisans(self.init_mean, self.init_sd)
        return self

    @classmethod
    def from_init(cls, init_mean, init_sd, **kwargs):
        assert len(init_mean) == len(init_sd)
        ret = cls(n = len(init_mean), init_mean = init_mean, init_sd = init_sd, **kwargs)
        ret.init_mean, ret.init_sd = ret._apply_partisans(init_mean, init_sd)
        return ret

    @classmethod
    def from_prior_params(cls, prior_params: PriorParams):
        init_mean, init_sd = prior_params.get()
        return GeneratedPriorParams(
            n=prior_params.n,
            init_mean=init_mean,
            init_sd=init_sd,
            partisans=prior_params.partisans,
            partisan_mean=prior_params.partisan_mean,
            partisan_sd=prior_params.partisan_sd,
            generated_from=prior_params
        )

from src.utils import UUID, uuid_field
@dataclass
class Simulation:
    _: KW_ONLY  # this is to avoid confusion, forces the user to use named args
    graph: Graph
    prior_params: PriorParams
    coin_params: CoinParams | None = None
    ai_params: AIParams | None = None
    discretised_bias: np.array = field(default_factory=lambda: DEFAULT_DISCRETISED_BIAS)
    metadata: dict = field(default_factory=dict)
    likelihood_generator: LikelihoodGenerator | None = None
    uuid: UUID = uuid_field()

    def __post_init__(self):
        n = self.prior_params.n
        assert self.graph.num_vertices() == n, f"graph n mismatches prior n ({self.graph.num_vertices()}, {n})"

        assert self.coin_params is not None or self.ai_params is not None, "either coin_params or ai_params must be provided"

        self.prior_means, self.prior_sds = self.prior_params.get()
        bias_mat = np.tile(self.discretised_bias, (n, 1))
        self.sim_params = SimParams(
            friendliness = friendliness_mat(self.graph),
            adjacency = adjacency_mat(self.graph),
            prior_distrs = init_prior_distrs(self.prior_means, self.prior_sds, bias_mat),
            coins = self.coins,
            partisans = self.prior_params.partisans,
            discretised_bias = self.discretised_bias,
            ai_params=self.ai_params,
            likelihood_generator=self.likelihood_generator,
        )
    
    @property
    def n(self):
        return self.prior_params.n

    @cached_property
    def coins(self):
        if self.coin_params:
            return self.coin_params.get()
        else:
            return self.ai_params.theta_ai
    
    @cached_property
    def true_bias(self):
        if self.coin_params:
            return self.coin_params.bias
        else:
            return self.ai_params.theta_ai

    @cached_property
    def partisan_prior(self):
        return self.prior_means[self.prior_params.partisans], self.prior_sds[self.prior_params.partisans]

    @cached_property
    def result(self):
        return run_simulation(self.sim_params)

    @cached_property
    def analyse(self, idx=0):
        from src.analyse.analyse import AnalyseSimulation, SimResults
        from src.analyse.results import parse_result

        return AnalyseSimulation(
            results = parse_result(SimResults(
                **(self.result._asdict() | dict(
                    coins = self.coins,
                    final_distr = self.result.distrs[-1],
                    initial_distr = self.sim_params.prior_distrs,
                    adjacency = self.sim_params.adjacency,
                    friendliness = self.sim_params.friendliness,
                ))
            )), 
            args = dict(
                size = self.sim_params.n,
                disruption = self.sim_params.partisans,
                bias = self.discretised_bias,
            ),
            sim_params = dict(
                asymptotic_learning_max_iters = self.sim_params.asymptotic_learning_max_iters,
                true_bias = self.true_bias,
            ),
            idx = idx
        )
    
    def get_other_result(self, field_name: str):
        if not hasattr(self.result, 'others'):
            raise AttributeError("self.result has no other results")
        
        if not isinstance(self.result.others, list):
            raise RuntimeError("self.result.other is not a list")
        
        if not self.result.others:
            return []
        
        if field_name not in self.result.others[0]:
            raise AttributeError(f"self.result.others entries have no field {field_name}")
    
        return [step_entry[field_name] for step_entry in self.result.others]
    
    @cached_property
    def asymptotic_per_agent(self) -> np.array:
        from src.analyse.dwell import asymptotic_per_agent
        return asymptotic_per_agent(self.result.distrs)

    @cached_property
    def dwells(self):
        asymp = self.asymptotic_per_agent
        
        # append 0 to the end of each list, so that the shape becomes (10, 10001)
        zeros_shape = asymp.shape[:-1] + (1,)
        asymptotic = np.concatenate((asymp, np.zeros(zeros_shape, dtype=int)), axis=1)

        dwell_agent_ids, dwell_indices = np.where(np.diff(asymptotic) < 0)
        dwell_times = np.array(asymptotic)[dwell_agent_ids, dwell_indices]
        # dwell_indices + 1 as we want to get the dwell at the peak, not the step before 
        # (note: np.diff makes shape smaller by 1)
        
        dwell_beliefs = np.array(self.result.mean_list)[dwell_indices, dwell_agent_ids]

        dwells = np.stack((dwell_agent_ids, dwell_indices, dwell_times, dwell_beliefs), axis=1)
        return dwells

    def dump(self, filename):
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filename):
        import pickle
        with open(filename, 'rb') as f:
            return pickle.load(f)