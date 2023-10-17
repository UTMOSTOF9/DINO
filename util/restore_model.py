
def restore_params(model, ckpt_params):
    model_params = model.state_dict()
    for k, v in model_params.items():
        if k in ckpt_params and v.shape == ckpt_params[k].shape:
            model_params[k] = ckpt_params[k]
        else:
            print(f"Warning: {k} is not in the checkpoint")
    model.load_state_dict(model_params, strict=True)
    return model