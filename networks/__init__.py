from networks.fsrcnn import FSRCNN


NETWORK_NAME_DICT = {
    'FSRCNN': FSRCNN,
}


def load_sr_network(network_name):
    if network_name in NETWORK_NAME_DICT.keys():
        net_cls = NETWORK_NAME_DICT[network_name]
    else:
        raise Exception(f'Check your network name, {network_name} is not in the following available networks: \n{NETWORK_NAME_DICT.keys()}')
    return net_cls
