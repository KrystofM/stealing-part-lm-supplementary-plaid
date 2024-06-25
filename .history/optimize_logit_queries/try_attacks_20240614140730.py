"""
THIS IS A TESTING FILE, for trying run_attack out on different strategies. Do not report results from this file.
"""

# %%
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

from run_attack import run_attack, make_proper
from askers.bias import bias_unexplored, bias_uncertain


data_sources = ["random", "llama-2-7b-chat", "llama-2-7b", "openai"]


def get_logit_vectors(data_source: str, n_trials: int, ntok: int) -> list[np.ndarray]:
    np.random.seed(1)
    random.seed(1)
    if "llama" in data_source:
        if data_source == "llama-2-7b":
            data = np.load(
                "../distribution_logits/data/meta-llama/Llama-2-7b-hf-last_token_logits.pkl",
                allow_pickle=True,
            )
        elif data_source == "llama-2-7b-chat":
            data = np.load(
                "../distribution_logits/data/meta-llama/Llama-2-7b-chat-hf-last_token_logits.pkl",
                allow_pickle=True,
            )
        else:
            raise NotImplementedError

        assert len(data.shape) == 2 and data.shape[1] == 32000

        data = data[: (n_trials * ntok) // data.shape[1] + 1]
        print(f"Using {data.shape[0]} rows of data")

        # rearrange the first and second coord randomly
        order_tokens = np.random.permutation(data.shape[1])
        data = data[:, order_tokens]
        order_rows = np.random.permutation(data.shape[0])
        data = data[order_rows, :]

        base_top_tokens = np.argmax(data, axis=1)

        # make batches of size ntok, all containing the base top token
        data_batches = []
        for i in range(data.shape[0]):
            j = 0

            # we skip the last batch
            while j + ntok - 1 < data.shape[1]:
                if j <= base_top_tokens[i] < j + ntok - 1:
                    batch = data[i, j : j + ntok]
                    j += ntok
                else:
                    batch = np.append(
                        data[i, j : j + ntok - 1], data[i, base_top_tokens[i]]
                    )
                    j += ntok - 1

                np.random.shuffle(batch)
                assert batch.shape == (ntok,)
                data_batches.append(batch)

        # now we have a list of batches, each of size ntok, containing the base top token
        assert len(data_batches) >= n_trials

        reals = data_batches[:n_trials]

    elif data_source == "openai":
        raise NotImplementedError

    else:
        raise NotImplementedError

    return reals


DATA_SOURCE = "llama-2-7b"
NTOK = 50
N_TRIALS = 10
ASSUMED_WIDTH = 40.0

if DATA_SOURCE == "random":
    np.random.seed(1)
    random.seed(1)
    reals = [np.random.uniform(0, 1, size=NTOK) for _ in range(N_TRIALS)]
    for i in range(len(reals)):
        base_top_token = np.argmax(reals[i])
        reals[i][base_top_token] = 1
else:
    reals = get_logit_vectors(DATA_SOURCE, N_TRIALS, NTOK)
    reals = [make_proper(real, assumed_width=ASSUMED_WIDTH) for real in reals]


IS_SORTED = True
if IS_SORTED:
    # sort s.t. 1 = x_0 >= x_1 >= ... >= x_{NTOK - 1}
    reals = [reals[i][np.argsort(-reals[i])] for i in range(len(reals))]
    print("Sorted reals:")
    for real in reals:
        print(real)

results = []
# Run the strategies and plot the errors
tolerance = 1e-5
error_type = "carlini_og_error"
real = reals[0]
for real in reals:
    # results.append(run_attack(real, tolerance, error_type, "every_one_over_n_uniform"))
    # results.append(run_attack(real, tolerance, error_type, "some_over_n", "iterate_constraints"))
    results.append(run_attack(real, tolerance, error_type, "start_one_over_n_normal", "iterate_constraints"))
    #results.append(run_attack(real, tolerance, error_type, "start_one_over_n_normal_V2", "iterate_constraints"))
    results.append(
        run_attack(
            real, tolerance, error_type, "simultaneous_binary_search", "iterate_constraints"
        )
    )
    results.append(run_attack(real, tolerance, error_type, "start_one_over_n", "iterate_constraints"))
    #results.append(run_attack(real, tolerance, error_type, "every_one_over_n_normal"))

    # results.append(run_attack(real, tolerance, error_type, "start_one_over_n_with_prior"))
    # results.append(run_attack(real, tolerance, error_type, "start_one_over_n", "iterate_constraints"))
    # results.append(run_attack(real, tolerance, error_type, "simultaneous_binary_search", "iterate_constraints"))
    # # results.append(
    # #     run_attack(real, tolerance, error_type, "start_one_over_n_merged", "iterate_constraints")
    # # )
    results.append(run_attack(real, tolerance, error_type, "normal_perfect", precision=1e-7))
    # results.append(run_attack(real, tolerance, error_type, "normal_distribution"))
    # results.append(
    #     run_attack(real, tolerance, error_type, "least_squares_all", "floyd_warshall")
    # )
    # results.append(run_attack(real, tolerance, error_type, "kde_distribution"))
    # generate 30 random permutations of the arrange of numbers from 0 to len(real)
    # orders = [np.random.permutation(len(real)) for _ in range(30)]
    # for order in orders:
    #     results.append(run_attack(real, tolerance, error_type, "normal_perfect", precision=0.01, order=order))
    # # results.append(run_attack(real, tolerance, error_type, "kde_distribution", kwargs={"plot": True}))
    # results.append(run_attack(real, tolerance, error_type, "normal_distribution", kwargs={"plot": True}))    
    # results.append(
    #     run_attack(
    #         real, tolerance, error_type, "simultaneous_binary_search", "iterate_constraints"
    #     )
    # )
# results.append(run_attack(real, tolerance, error_type, "normal_binary_search"))
# results.append(
#     run_attack(
#         real, tolerance, error_type, "simultaneous_binary_search", "iterate_constraints"
#     )
# )

# # %%
# results.append(
#     run_attack(
#         real, tolerance, error_type, "simultaneous_binary_search", "floyd_warshall"
#     )
# )
# # %%
# results.append(
#     run_attack(real, tolerance, error_type, "start_one_over_n", "floyd_warshall")
# )
# # %%
# results.append(
#     run_attack(
#         real, tolerance, error_type, "simultaneous_binary_search", "bellman_ford"
#     )
# )


# %%
# Things that don't work well:
# "uncertain", "iterate_constraints"
# "hypercube_actual_center"
#

# %%
# Example proper vector to run the attack on

# Collect results from each run_attack call

# alphas = np.arange(0.2, 0.9, 0.1)
# for alpha in alphas:
#     result = run_attack(
#         real=real,
#         tolerance=tolerance,
#         error_type=error_type,
#         query_strategy="unexplored",
#         bounds_computer="iterate_constraints",
#         method="linear_decay",
#         alpha=alpha,
#     )
#     results.append(result)

# alphas = [0.9, 0.95]
# for alpha in alphas:
#     result = run_attack(
#         real=real,
#         tolerance=tolerance,
#         error_type=error_type,
#         query_strategy="unexplored",
#         bounds_computer="iterate_constraints",
#         method="exp_decay",
#         alpha=alpha,
#     )
#     results.append(result)
# %%
# alphas = [0.9, 0.95]
# for alpha in alphas:
#     result = run_attack(
#         real=real,
#         tolerance=tolerance,
#         error_type=error_type,
#         query_strategy="unexplored",
#         bounds_computer="bellman_ford",
#         method="exp_decay",
#         alpha=alpha,
#     )
#     results.append(result)


# # %%
# alphas = np.arange(0.7, 1.11, 0.05)
# betas = np.arange(0.6, 0.9, 0.05)
# for alpha in alphas:
#     for beta in betas:
#         result = run_attack(
#             real=real,
#             tolerance=tolerance,
#             error_type=error_type,
#             query_strategy="mixed",
#             bounds_computer="iterate_constraints",
#             method="exp_decay",
#             alpha=alpha,
#             beta=beta,
#             callables=[bias_unexplored, bias_uncertain],
#             weights=[0.5, 0.5],
#         )
#         results.append(result)

# %%


##% Plot the worst case error
## This doesn't account for multiple types of errors we report; it's just for the carlini_og_error
def plot_worst_case_error(error_type, results):
    Q = max([len(result["errors"][error_type]) for result in results])
    errors = []
    for I in tqdm(range(Q)):
        errors.append(2 ** (-I * np.log2(NTOK) / NTOK))
    print(errors[-1])
    plt.plot(np.log(errors), label="worst case error")


# %%
# save errors
import pickle
from datetime import datetime
from pathlib import Path

Path("errors").mkdir(exist_ok=True)
now = datetime.now()
timestamp = now.strftime("%Y%m%d-%H%M%S")
filename = f"errors/{timestamp}.pickle"
with open(filename, "wb") as f:
    pickle.dump(results, f)
with open(filename, "rb") as f:
    results = pickle.load(f)


# Initialize a dictionary to store attack types and their corresponding colors
attack_colors = {}
color_cycle = plt.cm.get_cmap("tab10")  # Using a colormap with 10 distinct colors
used_labels = set() 

# Function to get color for each attack type
def get_color(attack_name):
    if attack_name not in attack_colors:
        # Assign a new color from the color cycle and increment the index
        attack_colors[attack_name] = color_cycle(len(attack_colors) % 10)  # Modulo to cycle through colors if more than 10 types
    return attack_colors[attack_name]

# %%
# plotting
ERROR_TO_PLOT = "carlini_og_error"
plt.figure(figsize=(17, 8))
for result in results:
    label = result["label"]
    color = get_color(label)
    # Check if the label has already been used
    if label not in used_labels:
        plt.plot(np.log(result["errors"][ERROR_TO_PLOT]), label=label, color=color)
        used_labels.add(label)
    else:
        plt.plot(np.log(result["errors"][ERROR_TO_PLOT]), color=color)

# plot_worst_case_error(error_type, results)
plt.xlabel("Iterations")
plt.ylabel("Error (log scale)")
plt.title(f"Error Over Iterations for Different Strategies, {ERROR_TO_PLOT}")
# legend out of the box
# larger resolution
plt.legend(
    bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0, fontsize="small"
)
plt.tight_layout()
fig = plt.gcf()
plt.show()
plt.draw()
fig.savefig("error_over_iterations.png", dpi=100)
# %%
# print steps
for result in results:
    print(result["label"])
    print(result["steps_to_tolerance"])


# %%
def get_steps_to_tolerance(
    result: dict, tolerance: float, error_type: str
) -> int | None:
    return next(
        (
            i
            for i, error in enumerate(result["errors"][error_type])
            if error < tolerance
        ),
        None,
    )


tolerance = 1e-6
for result in results:
    print(result["label"])
    print(get_steps_to_tolerance(result, tolerance, error_type))


# %%
print(len(reals[0]))