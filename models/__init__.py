from models.sr import SuperResolution


MODEL_NAME_DICT = {
    'sr': SuperResolution,
}


def load_model(model_name):
    if model_name in MODEL_NAME_DICT.keys():
        model_cls = MODEL_NAME_DICT[model_name]
    else:
        raise Exception(f'Check your network name, {model_name} is not in the following available networks: \n{MODEL_NAME_DICT.keys()}')
    return model_cls