from ..core import extend
from ..operations import *



def woodbury_iefvp_naive(
        vec,
        model,
        inputs,
        targets,
        loss_fn,
        damping=1e-5,
        data_average=True,
):
    """
    Calculate inverse-empirical Fisher vector product by using the Woodbury matrix identity
    """
    assert damping > 0, 'Damping value has to be positive.'

    with extend(model, OP_BATCH_GRADS):
        model.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets, reduction='sum')
        loss.backward()

        batch_g_all = []
        for _, batch_g in _module_batch_flatten_grads(model):
            batch_g_all.append(batch_g)
        grads = torch.cat(batch_g_all, dim=1).T  # (p, n)

    p, n = grads.shape
    if data_average:
        grads /= np.sqrt(n)
    assert vec.shape == (p,)
    gram = torch.matmul(grads.T, grads)  # (n, n)
    inv = torch.inverse(gram + torch.eye(n) * damping)  # (n, n)
    b = torch.matmul(inv, torch.matmul(grads.T, vec))

    return (vec - torch.matmul(grads, b)) / damping  # (p,)
