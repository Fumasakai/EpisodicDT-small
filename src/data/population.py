"""Fixed UE population shared by source episodes, model and generated scenarios."""
KINDS = ('car', 'walker', 'static')


def validate_population(population):
    if not isinstance(population, dict) or set(population) != {'min_ues', 'max_ues', 'counts'}:
        raise ValueError('population requires min_ues, max_ues and counts')
    lo, hi = population['min_ues'], population['max_ues']
    if type(lo) is not int or type(hi) is not int or not 1 <= lo <= hi:
        raise ValueError('Require 1 <= min_ues <= max_ues (integers)')
    return population_slots(population)


def population_slots(population, counts=None):
    counts = population['counts'] if counts is None else counts
    if not isinstance(counts, dict) or set(counts) != set(KINDS):
        raise ValueError('UE counts must specify car, walker and static')
    if any(type(v) is not int or v < 0 for v in counts.values()):
        raise ValueError('UE counts must be nonnegative integers')
    if not population['min_ues'] <= sum(counts.values()) <= population['max_ues']:
        raise ValueError('Total UE count is outside min_ues/max_ues')
    return [(kind, i+1) for kind in KINDS for i in range(counts[kind])]


def population_offsets(environment):
    """One fixed layout per batch. Legacy car_offsets remains readable."""
    import math
    if 'population' not in environment:
        offsets = {'car': environment.get('car_offsets', [[0, 0]]),
                   'walker': [[0, 0]], 'static': [[0, 0]]}
    else:
        validate_population(environment['population'])
        if 'car_offsets' in environment:
            raise ValueError('Use ue_offsets with population; remove legacy car_offsets')
        spacing = environment.get('ue_spacing_m', 10.)
        if isinstance(spacing, bool) or not isinstance(spacing, (int, float)) or not math.isfinite(spacing) or spacing < 0:
            raise ValueError('ue_spacing_m must be finite and nonnegative')
        specified = environment.get('ue_offsets', {})
        if not isinstance(specified, dict) or set(specified) - set(KINDS):
            raise ValueError('ue_offsets only supports car, walker and static')
        counts = environment['population']['counts']
        offsets = {kind: specified.get(kind, [[0., i*spacing] for i in range(counts[kind])]) for kind in KINDS}
        for kind in KINDS:
            if not isinstance(offsets[kind], list) or len(offsets[kind]) != counts[kind]:
                raise ValueError(f'ue_offsets/{kind} length must match population count')
    for kind, pairs in offsets.items():
        if not isinstance(pairs, list) or ('population' not in environment and not pairs):
            raise ValueError(f'Invalid offsets for {kind}')
        for pair in pairs:
            if (not isinstance(pair, (list, tuple)) or len(pair) != 2 or
                    any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in pair)):
                raise ValueError(f'{kind} offsets require finite [x,y] pairs')
    return offsets


def environment_slots(environment):
    offsets = population_offsets(environment)
    return [(kind, i+1) for kind in KINDS for i in range(len(offsets[kind]))]
