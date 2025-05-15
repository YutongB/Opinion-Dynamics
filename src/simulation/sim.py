from dataclasses import dataclass, field, KW_ONLY, asdict
from functools import cached_property
import numpy as np
from scipy.stats import binom, norm

from src.simulation.stat import normalize_distr
from src.analyse.results import SimResults

DEFAULT_MAX_STEPS = 10000
DEFAULT_DISCRETISED_BIAS = np.linspace(0, 1, 21)
CoinsList = list[int]

@dataclass
class AIParams:
    theta_ai: float = 0.8
    sigma_ai: float = 0.2
    max_steps: int = DEFAULT_MAX_STEPS

from abc import ABC, abstractmethod

@dataclass
class LikelihoodGenerator(ABC):
    n: int
    _: KW_ONLY
    discretised_bias: np.array = field(default_factory=lambda: DEFAULT_DISCRETISED_BIAS)

    def prepare(self):
        pass

    @abstractmethod
    def likelihood(self) -> tuple[np.array, dict]:
        ...

@dataclass
class SavedLikelihoodGenerator(LikelihoodGenerator):
    likelihoods: np.array
    # others: list[dict]  # currently unused, could use later

    def prepare(self):
        self.step = 0

    def likelihood(self) -> tuple[np.array, dict]:
        likelihood = self.likelihoods[self.step]
        self.step += 1
        # return likelihood, self.others
        return likelihood[:self.n], {}

@dataclass
class CoinsListLikelihoodGenerator(LikelihoodGenerator):
    coins: CoinsList
    tosses_per_iteration: int

    def prepare(self):
        self.step = 0

        self.likelihoods = [binom.pmf(num_heads, self.tosses_per_iteration, self.discretised_bias) for num_heads in range(self.tosses_per_iteration + 1)]

    def likelihood(self) -> tuple[np.array, dict]:
        coins = self.coins[self.step]
        self.step += 1
        return self.likelihoods[coins], {}
    
@dataclass
class AILikelihoodGenerator(LikelihoodGenerator):
    theta_ai: float
    sigma_ai: float

    def prepare(self):
        self.uses_ai_per_agent = binom.rvs(n=1, p=0.5, size=self.n)
        
    def likelihood(self) -> tuple[np.array, dict]:
        uses_ai_per_agent = self.uses_ai_per_agent
        ai_score_per_agent = np.clip(
            np.round(norm.rvs(loc=self.theta_ai, scale=self.sigma_ai, size=self.n), 2),
            0, 1) # (n,)

        # Code below is equivalent to the following code:
        # likelihood = [norm.pdf(p.discretised_bias, loc=ai_score, scale=sigma_ai) if agent_uses_ai else np.ones_like(p.discretised_bias)
        #               for ai_score, agent_uses_ai in zip(ai_score_per_agent, uses_ai_per_agent)]

        true_likelihoods = norm.pdf(self.discretised_bias[:, None], loc=ai_score_per_agent, scale=self.sigma_ai) # (21, n)
        likelihood = np.ones((len(self.discretised_bias), self.n)) # (21, n)
        likelihood[:, uses_ai_per_agent == 1] = true_likelihoods[:, uses_ai_per_agent == 1] # (21, n)

        # likelihood dimension must be (n, 21) to allow element-wise multiplication with prior_distr, which is also (n, 21), resulting in a (n, 21) array
        # So we transpose.
        return likelihood.T, dict(uses_ai_per_agent=uses_ai_per_agent, ai_score_per_agent=ai_score_per_agent)

@dataclass
class MaskedAILikelihoodGenerator(AILikelihoodGenerator):
    num_using_ai: int

    def prepare(self):
        arr = np.array([1] * self.num_using_ai + [0] * (self.n - self.num_using_ai))
        self.uses_ai_per_agent = arr

@dataclass(frozen=True)
class SimParams:
    # n x n matrix
    friendliness: np.array
    # n x n matrix
    adjacency: np.array
    # n x b matrix
    prior_distrs: np.array
    """list of coins"""
    coins: np.array
    """indices of partisans"""
    partisans: list[int]
    # length b vector
    discretised_bias: np.array
    # TODO: likelihood: list[np.array]
    """μ / μ_i in the paper"""
    learning_rate = 0.25
    asymptotic_learning_max_iters = 99
    break_on_asymptotic_learning = False
    """number of coin tosses performed on each simulation timestep"""
    tosses_per_iteration = 1

    ai_params: AIParams | None = None
    likelihood_generator: LikelihoodGenerator | None = None

    stat_names: set[str] = field(default_factory=lambda: {'mean', 'std', 'asymptotic'})

    @cached_property
    def n(self) -> int:
        return len(self.friendliness)
    
    @cached_property
    def max_steps(self) -> int:
        if self.ai_params is None:
            return len(self.coins)
        else:
            return self.ai_params.max_steps

    @cached_property
    def bias_mat(self):
        return np.tile(self.discretised_bias, (self.n, 1))

    def __post_init__(self):
        assert self.n == len(self.prior_distrs), f"prior_distrs shape[0] must be n ({self.n}, {len(self.prior_distrs)})"
        assert len(self.prior_distrs.shape) == 2, f"prior_distrs must be a 2d array, got {self.prior_distrs.shape}"
        assert len(self.discretised_bias) == self.prior_distrs.shape[1]

        if self.likelihood_generator is None:
            if self.ai_params is None:
                likelihood_generator = CoinsListLikelihoodGenerator(coins=self.coins, tosses_per_iteration=self.tosses_per_iteration, n=self.n, discretised_bias=self.discretised_bias)
            else:
                likelihood_generator = AILikelihoodGenerator(theta_ai=self.ai_params.theta_ai, sigma_ai=self.ai_params.sigma_ai, n=self.n, discretised_bias=self.discretised_bias)
            # NOTE: self is frozen, so we can't set attributes directly
            object.__setattr__(self, 'likelihood_generator', likelihood_generator)

    # FIXME: Do these really belong here?

    def mean_distr(self, distrs):
        return np.sum(distrs * self.bias_mat, axis=1)
    
    def std_distr(self, distrs, mean):
        mean = mean[:, None]
        return np.sqrt(np.sum(np.square(self.bias_mat - mean) * distrs, axis=1))


def run_simulation(params: SimParams) -> SimResults:
    """
    max_steps (T in the paper) - maximum number of steps to run the simulation
    """
    p = params
    assert p.n > 1
    
    distrs = []
    means = []
    asymptotic = []
    iters_asymptotic_learning = 0
    iters_asymptotic_learning_agents = [0] * p.n
    prior_distr = p.prior_distrs.copy()
    distrs.append(prior_distr)

    p.likelihood_generator.prepare()
    all_others = []

    # steps 2-3 of Probabilistic automaton, until t = T
    steps = 0
    for step in range(p.max_steps):
        steps = step + 1

        likelihood, others = p.likelihood_generator.likelihood()
        all_others.append(others)

        # FIXME: add step_fn to SimParams so we don't need to comment this out

        posterior = step_simulation(
            prior_distr=prior_distr, 
            friendliness=p.friendliness,
            likelihood=likelihood,
            learning_rate=p.learning_rate,
            partisans=p.partisans,
        )

        # posterior = step_simulation_negative_reinforcement(prior_distr, friendliness=p.friendliness, likelihood=likelihood)

        means.append(p.mean_distr(posterior))
        distrs.append(posterior)
        
        # system reaches asymptotic learning when all agents reach asymptotic learning   
        largest_change = np.max(np.abs(posterior - prior_distr), axis=1)
        largest_peak = 0.01 * np.max(prior_distr, axis=1)
        agent_is_asymptotic = largest_change < largest_peak  # one element for each agent
        # iters_asymptotic_learning_agents = np.where(agent_is_asymptotic, iters_asymptotic_learning_agents + 1, 0)
        for i, agent_asymptotic in enumerate(agent_is_asymptotic):  # loop over each agent
            if agent_asymptotic:
                iters_asymptotic_learning_agents[i] += 1
            else:
                iters_asymptotic_learning_agents[i] = 0
        
        # for the whole system...
        if np.all(agent_is_asymptotic):
            iters_asymptotic_learning += 1
        else:
            iters_asymptotic_learning = 0
        asymptotic.append(iters_asymptotic_learning)

        if p.break_on_asymptotic_learning and iters_asymptotic_learning == p.asymptotic_learning_max_iters:
            break

        prior_distr = posterior.copy()

    # agent_is_asymptotic = np.array(iters_asymptotic_learning_agents) >= asymptotic_learning_max_iters
    agent_is_asymptotic = np.array(iters_asymptotic_learning_agents)

    return SimResults(steps=steps,
                    asymptotic=asymptotic,
                    agent_is_asymptotic=agent_is_asymptotic,
                    coins=None,
                    mean_list=means,
                    std_list=None,
                    final_distr=None,
                    initial_distr=None,
                    adjacency=None,
                    friendliness=None,
                    distrs=distrs,
                    others=all_others)

EPSILON = 0

def step_simulation(prior_distr, *, friendliness, likelihood: list[np.ndarray], learning_rate, partisans):
    n, bias_samples = prior_distr.shape

    posterior_distr = normalize_distr(likelihood * prior_distr)  # (eqn 1)

    # Rows: node i's posterior distribution.  Cols: posterior distr evaluated at theta
    # Need to change partisan's belief after cointoss before blending
    posterior_distr[partisans] = prior_distr[partisans]

    # Mix the opinions of each node with respective neighbours (eq 3,4)
    xi = np.broadcast_to(posterior_distr, (n, n, bias_samples)).transpose((1, 0, 2))
    xj = np.broadcast_to(posterior_distr, (n, n, bias_samples))
    # calculate the difference in belief for node i to node j
    diff_in_belief = ((xj - xi).T * friendliness).T
    summation = np.sum(diff_in_belief, axis=1)
    # sums over the columns
    divisor = np.reciprocal(np.sum(np.abs(friendliness), axis=1))

    avg_dist_in_belief = summation * divisor[:, None]

    bayes_update_max_RHS = posterior_distr + avg_dist_in_belief * learning_rate

    bayes_update_max_LHS = np.broadcast_to(EPSILON, (n, bias_samples))
    bayes_update = np.amax(np.array([bayes_update_max_LHS, bayes_update_max_RHS]), axis=0)

    next_prior_distr = normalize_distr(bayes_update)
    next_prior_distr[partisans] = prior_distr[partisans]

    return next_prior_distr

# FIXME: pass through rng and discretised_bias
# FIXME: test that not normalising the likelihoods doesn't break anything by comparing with same setup and rng
def coin_toss_with_sampled_bias_and_bayesian_update(prior_distr,
                                                    distr_to_sample,
                                                    rng = np.random.default_rng(),
                                                    discretised_bias=DEFAULT_DISCRETISED_BIAS,
                                                    relationship = 1):
    #      sample a bias from posterior distribution
    bias = np.random.choice(a=discretised_bias, p=distr_to_sample)

    #      make a biased coin flip with the sampled bias
    if relationship == 1:
        coin = rng.binomial(n=1, p=bias)
    elif relationship == -1:
        coin = int(not rng.binomial(n=1, p=bias))
    else:
        raise ValueError("relationship must be 1 or -1")
    #      update bayesian with coin toss
    likelihoods = [binom.pmf(num_heads, 1, discretised_bias) for num_heads in range(2)]
    likelihood = likelihoods[coin]
    
    return likelihood * prior_distr

def step_simulation_negative_reinforcement(prior_distr, *, friendliness, likelihood: list[np.ndarray]):
    n, _ = prior_distr.shape

    posterior_distr = likelihood * prior_distr  # (eqn 1)

    for i in range(n):
        for j in range(n):
            if friendliness[i, j] == 1:
                posterior_distr[i] = coin_toss_with_sampled_bias_and_bayesian_update(posterior_distr[i], prior_distr[j])
                
            elif friendliness[i, j] == -1:
                posterior_distr[i] = coin_toss_with_sampled_bias_and_bayesian_update(posterior_distr[i], prior_distr[j])
        
    posterior_distr = normalize_distr(posterior_distr)
    return posterior_distr

def normalise_distr_1d(distr):
    return distr / np.sum(distr)

# It's slow because it normalises all the time
def step_simulation_negative_reinforcement_SLOW(prior_distr, *, friendliness, likelihood: list[np.ndarray]):
    n, _ = prior_distr.shape

    posterior_distr = normalize_distr(likelihood * prior_distr)  # (eqn 1)

    for i in range(n):
        for j in range(n):
            if friendliness[i, j] == 1:
                posterior_distr[i] = normalise_distr_1d(
                    coin_toss_with_sampled_bias_and_bayesian_update(posterior_distr[i], prior_distr[j]))
                
            elif friendliness[i, j] == -1:
                posterior_distr[i] = normalise_distr_1d(
                    coin_toss_with_sampled_bias_and_bayesian_update(posterior_distr[i], prior_distr[i]))
        
    return posterior_distr
