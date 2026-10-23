"""Bounded, ordered execution of independent external simulator jobs."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED


def evaluate_jobs(sim, jobs, workers=1, progress=None):
    """Return results in input order; optionally report (completed, total).

    Progress runs in the caller, including a heartbeat every 30s while waiting.
    Model sampling and autograd stay in the caller; workers only run simulators.
    """
    if type(workers) is not int or workers < 1:
        raise ValueError('simulation_workers must be a positive integer')
    jobs = list(jobs)
    if progress:
        progress(0, len(jobs))
    if workers == 1 and progress is None:
        return [sim.try_evaluate(*args) for args in jobs]
    results = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(sim.try_evaluate, *args): i for i, args in enumerate(jobs)}
        completed = 0
        while pending:
            done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                results[pending.pop(future)] = future.result()
                completed += 1
            if progress:
                progress(completed, len(jobs))
    return results
