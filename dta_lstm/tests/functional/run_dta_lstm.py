import pprint

import numpy as np
import pandas as pd

from dta_lstm.network.dta_lstm import DTA_LSTM
from dta_lstm.network.config import DEFAULTS

np.random.seed(42)


OPENSTACK_SEQUENCE = [
    ('keystone', 1.0, 'authenticate user with credentials and generate auth-token'),
    ('nova-api', 1.0, 'get user request and sends token to Keystone for validation'),
    ('keystone', 0.2, 'validate the token if not in cache'),
    ('nova-api', 1.0, 'starts VM creation process'),
    ('nova-database', 1.0, 'create initial database entry for new VM'),
    ('nova-scheduler', 1.0, 'locate an appropriate host using filters and weights'),
    ('nova-database', 1.0, 'execute query'),
    ('nova-compute', 1.0, 'dispatches request'),
    ('nova-conductor', 1.0, 'gets instance information'),
    ('nova-compute', 1.0, 'get image URI from image service'),
    ('glance-api', 1.0, 'get image metadata'),
    ('nova-compute', 1.0, 'setup network'),
    ('neutron-server', .5, 'allocate and configure network IP address'),
    ('nova-compute', 1.0, 'setup volume'),
    ('cinder-api', .75, 'attach volume to the instance or VM'),
    ('nova-compute', 1.0, 'generates data for the hypervisor driver')
]


def create_coders():

    encoder = {f: i + DEFAULTS['padding_symbol'] + 1
               for i, f in enumerate(sorted(set(function for function, _, _ in OPENSTACK_SEQUENCE)))}
    decoder = {i: f for i, f in encoder.items()}

    return encoder, decoder


def generate_traces(encoder, n=10, base_id=0):

    raw_traces = [[trace_id, [function for function, p, _ in OPENSTACK_SEQUENCE if np.random.random(size=1) < p]]
                  for trace_id in range(base_id, base_id + n)]

    encoded_traces = [[trace_id, [encoder[f] for f in functions]] for trace_id, functions in raw_traces]

    return raw_traces, encoded_traces


def mutate_traces(encoded_traces, mode=None, pos=0):
    mode = mode or 'del'
    if mode == 'del':
        return [[trace_id, [function for i, function in enumerate(functions) if i != pos]]
                for trace_id, functions in encoded_traces]
    if mode == 'replace':
        return [[trace_id, [function if i != pos else 1 for i, function in enumerate(functions)]]
                for trace_id, functions in encoded_traces]

    print('Unknown mode')

    return encoded_traces


def run_lstm(train, test):
    tm = DTA_LSTM()
    tm.fit(train)
    return tm.predict(test)


if __name__ == "__main__":

    # Create an encoder and decoder to transforms function names into integers
    encoder, decoder = create_coders()
    pprint.pprint(encoder)

    # Generate n=1000 traces to be used for training the model
    raw_traces, encoded_traces = generate_traces(encoder, n=100)
    pprint.pprint(raw_traces[0])
    # pprint.pprint(encoded_traces)
    df_train_traces = pd.DataFrame(encoded_traces,
                                   columns=['trace_id', DEFAULTS['span_list_col_name']]).set_index('trace_id')

    # Generate n=1 traces to be for testing the model
    _, encoded_traces = generate_traces(encoder, n=1)
    # Replace a function id at position pos=3
    test_traces = mutate_traces(encoded_traces, mode='replace', pos=3)
    _, encoded_traces = generate_traces(encoder, n=1, base_id=1)
    # pprint.pprint(test_traces)

    # Create the set of traces to test: one anomalous trace and a normal trace
    test_traces.extend([encoded_traces[0]])
    pprint.pprint(test_traces)

    df_test_traces = pd.DataFrame(test_traces,
                                  columns=['trace_id', DEFAULTS['span_list_col_name']]).set_index('trace_id')

    print(df_train_traces)
    print(df_test_traces)

    # Run the lstm algorithm
    for trace_id, rank_prob in enumerate(run_lstm(df_train_traces, df_test_traces)):
        errors_at = [i for i, (rank, _) in enumerate(rank_prob) if rank > 3]
        print('trace_id', trace_id, 'indices', list(errors_at))


#                                                   span_list
# trace_id
# 0               [2, 4, 4, 8, 7, 5, 6, 5, 1, 5, 3, 5, 0, 5]
#        [0  2  4  2  X4 7  8  7  5  6  5  1  5  3  5  0  5]
#              [2, 4, 4, X, 8, 7, 5, 6, 5, 1, 5, 3, 5, 0, 5]
#
#                                                   span_list
# trace_id
# 0               [2, 4, 4, 7, 8, 7, 5, 6, 5, 1, 5, 5, 0, 5]
# 1               [2, 4, 4, 7, 8, 7, 5, 6, 5, 1, 5, 5, 0, 5]
# 2            [2, 4, 4, 7, 8, 7, 5, 6, 5, 1, 5, 3, 5, 0, 5]
# 3               [2, 4, 4, 7, 8, 7, 5, 6, 5, 1, 5, 3, 5, 5]
# 4            [2, 4, 2, 4, 7, 8, 7, 5, 6, 5, 1, 5, 5, 0, 5]
# 5                  [2, 4, 4, 7, 8, 7, 5, 6, 5, 1, 5, 5, 5]
# 6         [2, 4, 2, 4, 7, 8, 7, 5, 6, 5, 1, 5, 3, 5, 0, 5]
# 7               [2, 4, 4, 7, 8, 7, 5, 6, 5, 1, 5, 3, 5, 5]
# 8               [2, 4, 4, 7, 8, 7, 5, 6, 5, 1, 5, 5, 0, 5]
# 9               [2, 4, 4, 7, 8, 7, 5, 6, 5, 1, 5, 5, 0, 5]
