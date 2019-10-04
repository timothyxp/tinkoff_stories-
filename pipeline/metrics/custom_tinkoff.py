

def tinkoff_custom(prediction, target):
    weight_event = {
        0: -10,
        1: -0.1,
        2: 0.1,
        3: 0.5
    }

    weights = [weight_event[event] for event in prediction]

    answer = sum([weight * targ for weight, targ in zip(weights, target)])

    best_weight = {
        0: -1,
        1: -1,
        2: 1,
        3: 1
    }

    best = sum([best_weight[targ] * weight_event[targ] for targ in target])

    return float(answer) / best
