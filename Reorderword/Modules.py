import tensorflow as tf
from torch.nn import Transformer
def EmbedWord(length_of_word, dim_vector_embedding=64):
    dimension = dim_vector_embedding
    # j = np.arange(0, dimension)
    j = tf.range(0, dimension, dtype="float32")
    # i = np.arange(1, length_of_word + 1)
    i = tf.range(1, length_of_word + 1, dtype="float32")
    odd_mask = j % 2 == 1
    odd_i_mask = tf.expand_dims(odd_mask, axis = 0)
    j_mask = tf.expand_dims(j, axis = 0)
    result = tf.where(odd_i_mask,
                    tf.cos(i[:, tf.newaxis] * tf.pow(1/10000, (j_mask - 1) / dimension)), 
                    tf.sin(i[:, tf.newaxis] * tf.pow(1/10000, j_mask / dimension))
                    )
    return result
    # b1 = np.random.randint(10, size=(len(i), len(j)))  # Example random array b1

    # Create masks for odd and even elements of j
    # odd_mask = j % 2 == 1

    # Create masks for broadcasting
    # odd_i_mask = np.expand_dims(odd_mask, axis=0)
    # j_mask = np.expand_dims(j, axis=0)

    # # Add i + j for odd elements of j and i - j for even elements of j
    # result = np.where(
    #     odd_i_mask,
    #     np.cos(i[:, np.newaxis] * np.power(1 / 10000, (j_mask - 1) / dimension)),
    #     np.sin(i[:, np.newaxis] * np.power(1 / 10000, j_mask / dimension)),
    # )
    # return result

def Scale_Dot_Attention(queries,
                        keys,
                        dv):
    
    
