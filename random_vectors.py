import numpy as np
import argparse


parser = argparse.ArgumentParser(description="simulated herding bound values")
parser.add_argument("--d", type=int, required=True, help="worker model params")
parser.add_argument("--m", type=int, required=True,
                    help="number of examples per worker")
parser.add_argument("--n", type=int, required=True, help="number of worker")
parser.add_argument("--r", type=int, default=5, help="number of trials")
parser.add_argument("--round", type=int, default=15,
                    help="number of reorder rounds")
args = parser.parse_args()

print(vars(args))
m, n, d = args.m, args.n, args.d
mn = m * n

rounds = args.round


def herding_bound(vecs):
    return np.maximum.accumulate(np.linalg.norm(np.cumsum(vecs, 0), ord=float('inf'), axis=1))[-1]


def parallel_herding_bound(vecs):
    # vecs: m, n, d
    return np.maximum.accumulate(np.linalg.norm(np.cumsum(np.sum(vecs, axis=1), axis=0), ord=float('inf'), axis=1))[-1]


def reorder_from_signs(signs, single_vecs):
    next_vecs = np.empty_like(single_vecs)

    pos_half = np.where(signs == +1)
    pos_half_size = len(pos_half[0])
    next_vecs[:pos_half_size] = single_vecs[pos_half]

    neg_half = np.where(signs == -1)
    neg_half_size = len(neg_half[0])
    next_vecs[pos_half_size:] = single_vecs[neg_half][::-1]

    assert pos_half_size + neg_half_size == len(single_vecs)

    return next_vecs


def vanilla_balance(single_vecs):
    # single_vecs: mn, d
    run_sum = np.zeros_like(single_vecs[0])
    signs = np.zeros((len(single_vecs),), dtype=np.int8)
    for i, vec in enumerate(single_vecs):
        if np.linalg.norm(run_sum + vec, ord=2) <= np.linalg.norm(run_sum - vec, ord=2):
            signs[i] = +1
            run_sum += vec
        else:
            signs[i] = -1
            run_sum -= vec
    return signs


def vanilla_reorder_multiround(single_vecs, round):
    herding_bounds = []
    # single_vecs: mn, d
    herding_bounds.append(herding_bound(single_vecs))  # zero round
    for _ in range(round):
        signs = vanilla_balance(single_vecs)
        single_vecs = reorder_from_signs(signs, single_vecs)
        herding_bounds.append(herding_bound(single_vecs))
    return herding_bounds


def independent_parallel_balance(vecs):
    # vecs: m, n, d
    m, n, d = vecs.shape
    signs = np.zeros((m, n), dtype=np.int8)
    for j in range(n):
        signs_j = vanilla_balance(vecs[:, j, :])
        signs[:, j] = signs_j
    return signs


def independent_parallel_reorder(signs, vecs):
    n = vecs.shape[1]
    next_vecs = np.empty_like(vecs)
    for j in range(n):
        next_vecs[:, j, :] = reorder_from_signs(signs[:, j], vecs[:, j, :])
    return next_vecs


def independent_parallel_reorder_multiround(vecs, round):
    herding_bounds = []
    # vecs: m, n, d
    herding_bounds.append(parallel_herding_bound(vecs))  # zero round
    for _ in range(round):
        signs = independent_parallel_balance(vecs)
        vecs = independent_parallel_reorder(signs, vecs)
        herding_bounds.append(parallel_herding_bound(vecs))
    return herding_bounds


def parallel_pair_balance_and_reorder(vecs):
    # vecs: m, n, d
    m, n, d = vecs.shape
    mn = m * n
    run_sum = np.zeros((d,))

    next_epoch_vecs = np.empty_like(vecs)
    left, right = 0, m - 1

    for i in range(0, m, 2):
        pair_diff = vecs[i, :, :] - vecs[i + 1, :, :]  # n, d
        for j in range(n):
            if np.linalg.norm(run_sum + pair_diff[j], ord=2) <= np.linalg.norm(run_sum - pair_diff[j], ord=2):
                next_epoch_vecs[left, j] = vecs[i, j]  # +1
                next_epoch_vecs[right, j] = vecs[i + 1, j]  # -1
                run_sum += pair_diff[j]
            else:
                next_epoch_vecs[right, j] = vecs[i, j]  # -1
                next_epoch_vecs[left, j] = vecs[i + 1, j]  # +1
                run_sum -= pair_diff[j]
        left += 1
        right -= 1

    return next_epoch_vecs


def parallel_pair_balance_reorder_multiround(vecs, round):
    herding_bounds = []
    # vecs: m, n, d
    herding_bounds.append(parallel_herding_bound(vecs))
    for _ in range(round):
        vecs = parallel_pair_balance_and_reorder(vecs)
        herding_bounds.append(parallel_herding_bound(vecs))
    return herding_bounds


def centralized_pair_balance_and_reorder(vecs):
    # vecs: mn d
    mn, d = vecs.shape
    run_sum = np.zeros((d,))

    next_epoch_vecs = np.empty_like(vecs)
    left, right = 0, mn - 1

    for i in range(0, mn, 2):
        pair_diff = vecs[i, :] - vecs[i + 1, :]  # d
        if np.linalg.norm(run_sum + pair_diff, ord=2) <= np.linalg.norm(run_sum - pair_diff, ord=2):
            next_epoch_vecs[left] = vecs[i]  # +1
            next_epoch_vecs[right] = vecs[i + 1]  # -1
            run_sum += pair_diff
        else:
            next_epoch_vecs[right] = vecs[i]  # -1
            next_epoch_vecs[left] = vecs[i + 1]  # +1
            run_sum -= pair_diff
        left += 1
        right -= 1

    return next_epoch_vecs


def centralized_pair_balance_reorder_multiround(vecs, round):
    herding_bounds = []
    # vecs: mn, d
    herding_bounds.append(herding_bound(vecs))
    for _ in range(round):
        vecs = centralized_pair_balance_and_reorder(vecs)
        herding_bounds.append(herding_bound(vecs))
    return herding_bounds


def independent_pair_balance_reorder_multiround(vecs, round):
    herding_bounds = []
    # vecs: m, n, d
    herding_bounds.append(parallel_herding_bound(vecs))  # zero round
    for _ in range(round):
        for i in range(n):
            vecs[:, i, :] = centralized_pair_balance_and_reorder(vecs[:, i, :])
        herding_bounds.append(parallel_herding_bound(vecs))
    return herding_bounds


vanilla_herding_values = []
independent_herding_values = []
order_server_herding_values = []
centralized_pair_balance_herding_values = []
independent_pair_balance_herding_values = []

for i in range(args.r):
    vecs = np.random.RandomState(i).randn(mn, d)
    vecs = vecs / np.linalg.norm(vecs, ord=2, axis=1).reshape(mn, 1)
    vecs = vecs.reshape(m, n, d)
    vecs -= np.expand_dims(vecs.mean(axis=0), 0)

    vanilla_vecs = vecs.reshape(mn, d)
    vanilla_herding_values.append(
        vanilla_reorder_multiround(vanilla_vecs, rounds))
    independent_herding_values.append(
        independent_parallel_reorder_multiround(vecs, rounds))
    order_server_herding_values.append(
        parallel_pair_balance_reorder_multiround(vecs, rounds))
    centralized_pair_balance_herding_values.append(
        centralized_pair_balance_reorder_multiround(vanilla_vecs, rounds))
    independent_pair_balance_herding_values.append(
        independent_pair_balance_reorder_multiround(vecs, rounds)
    )
    del vecs
