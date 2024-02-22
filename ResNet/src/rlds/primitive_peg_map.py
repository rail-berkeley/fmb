import tensorflow as tf

FMB_PRIMITIVE_LIST =  ['grasp', 'place_on_fixture', 'regrasp', 'insert', 'rotate']
FMB_PRIMITIVE_TO_ID_DICT = {primitive: i for i, primitive in enumerate(FMB_PRIMITIVE_LIST)}

FMB_PEG_LIST = [1,2,3]
FMB_PEG_TO_ID_DICT = {peg: i for i, peg in enumerate(FMB_PEG_LIST)}

TF_CONSTANT_FMB_PRIMITIVE_LIST = tf.constant(FMB_PRIMITIVE_LIST)
def action_to_primitive(action):
    return action // len(FMB_PEG_LIST)
def action_to_peg(action):
    return action % len(FMB_PEG_LIST)

def tf_primitive_to_primitive_id(primitive):
    # Tensor of object type string
    primitive = tf.reshape(primitive, [-1, 1])
    match = tf.equal(primitive, TF_CONSTANT_FMB_PRIMITIVE_LIST)
    match = tf.argmax(tf.cast(match, tf.int32), axis=-1)
    return match

