from networks.fsrcnn import FSRCNN


NETWORK_NAME_DICT = {
    'FSRCNN': FSRCNN,
}


def print_network(model):
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('Network [%s] was created. Total number of parameters: %.1f kelo. '
          'To see the architecture, do print(network).'
          % (model.__class__.__name__, num_params / 1000))


def load_sr_network(network_name):
    if network_name in NETWORK_NAME_DICT.keys():
        net_cls = NETWORK_NAME_DICT[network_name]
    else:
        raise Exception(f'Check your network name, {network_name} is not in the following available networks: \n{NETWORK_NAME_DICT.keys()}')
    return net_cls
