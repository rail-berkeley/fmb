import dataclasses
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from flax.core import FrozenDict

class JaxRNG(object):
    """ A convenient stateful Jax RNG wrapper. Can be used to wrap RNG inside
        pure function.
    """

    @classmethod
    def from_seed(cls, seed):
        return cls(jax.random.PRNGKey(seed))

    def __init__(self, rng):
        self.rng = rng

    def __call__(self, keys=None):
        if keys is None:
            self.rng, split_rng = jax.random.split(self.rng)
            return split_rng
        elif isinstance(keys, int):
            split_rngs = jax.random.split(self.rng, num=keys + 1)
            self.rng = split_rngs[0]
            return tuple(split_rngs[1:])
        else:
            split_rngs = jax.random.split(self.rng, num=len(keys) + 1)
            self.rng = split_rngs[0]
            return {key: val for key, val in zip(keys, split_rngs[1:])}


def wrap_function_with_rng(rng):
    """ To be used as decorator, automatically bookkeep a RNG for the wrapped function. """
    def wrap_function(function):
        def wrapped(*args, **kwargs):
            nonlocal rng
            rng, split_rng = jax.random.split(rng)
            return function(split_rng, *args, **kwargs)
        return wrapped
    return wrap_function


def init_rng(seed):
    global jax_utils_rng
    jax_utils_rng = JaxRNG.from_seed(seed)


def next_rng(*args, **kwargs):
    global jax_utils_rng
    return jax_utils_rng(*args, **kwargs)


def extend_and_repeat(tensor, axis, repeat):
    return jnp.repeat(jnp.expand_dims(tensor, axis), repeat, axis=axis)


def mse_loss(val, target):
    return jnp.mean(jnp.square(val - target))


def value_and_multi_grad(fun, n_outputs, argnums=0, has_aux=False):
    def select_output(index):
        def wrapped(*args, **kwargs):
            if has_aux:
                x, *aux = fun(*args, **kwargs)
                return (x[index], *aux)
            else:
                x = fun(*args, **kwargs)
                return x[index]
        return wrapped

    grad_fns = tuple(
        jax.value_and_grad(select_output(i), argnums=argnums, has_aux=has_aux)
        for i in range(n_outputs)
    )
    def multi_grad_fn(*args, **kwargs):
        grads = []
        values = []
        for grad_fn in grad_fns:
            (value, *aux), grad = grad_fn(*args, **kwargs)
            values.append(value)
            grads.append(grad)
        return (tuple(values), *aux), tuple(grads)
    return multi_grad_fn


@jax.jit
def batch_to_jax(batch):
    return jax.tree_util.tree_map(jax.device_put, batch)


def collect_jax_metrics(metrics, names, prefix=None):
    collected = {}
    for name in names:
        if name in metrics:
            collected[name] = jnp.mean(metrics[name])
    if prefix is not None:
        collected = {
            '{}/{}'.format(prefix, key): value for key, value in collected.items()
        }
    return collected


def flatten_tree(xs, is_leaf=None, sep=None):
    """ A stronger version of flax.traverse_util.flatten_dict, supports
        dict, tuple, list and TrainState. Tuple and list indices will be
        converted to strings.
    """
    tree_node_classes = (FrozenDict, dict, tuple, list, TrainState)
    if not isinstance(xs, tree_node_classes):
        ValueError('fUnsupported node type: {type(xs)}')

    def _is_leaf(prefix, fx):
        if is_leaf is not None:
            return is_leaf(prefix, xs)
        return False

    def _key(path):
        if sep is None:
            return path
        return sep.join(path)

    def _convert_to_dict(xs):
        if isinstance(xs, (FrozenDict, dict)):
            return xs
        elif isinstance(xs, (tuple, list)):
            return {f'{i}': v for i, v in enumerate(xs)}
        elif isinstance(xs, TrainState):
            output = {}
            for field in dataclasses.fields(xs):
                if 'pytree_node' not in field.metadata or field.metadata['pytree_node']:
                    output[field.name] = getattr(xs, field.name)
            return output
        else:
            raise ValueError('fUnsupported node type: {type(xs)}')

    def _flatten(xs, prefix):
        if not isinstance(xs, tree_node_classes) or _is_leaf(prefix, xs):
            return {_key(prefix): xs}

        result = {}
        is_empty = True
        for (key, value) in _convert_to_dict(xs).items():
            is_empty = False
            path = prefix + (key, )
            result.update(_flatten(value, path))
        return result

    return _flatten(xs, ())


def named_tree_map(f, tree, is_leaf=None, sep=None):
    """ An extended version of jax.tree_util.tree_map, where the mapped function
        f takes both the name (path) and the tree leaf as input.
    """
    flattened_tree = flatten_tree(tree, is_leaf=is_leaf, sep=sep)
    id_to_name = {id(val): key for key, val in flattened_tree.items()}
    def map_fn(leaf):
        name = id_to_name[id(leaf)]
        return f(name, leaf)
    return jax.tree_util.tree_map(map_fn, tree)


def discrete_encode(codebook, data):
    return jax.vmap(
        lambda x: jnp.argmin(jnp.linalg.norm(codebook - x, axis=-1)),
    )(data)


def discrete_decode(codebook, data):
    return codebook[data, :]