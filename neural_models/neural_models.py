
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *


def MemN2N_model(mem_dim=32, num_hops=5, n_protein_properties=5, lr=1e-5):
    # n_protein_properties: ks, charges, masses, y_eqs, freqs

    # protein_properties = [Input(shape=(None, 1), name='input_{}'.format(i)) for i in range(n_protein_properties - 1)]
    protein_properties = [
        Input(shape=(None, 1), name='ks'),
        Input(shape=(None, 1), name='charges'),
        Input(shape=(None, 1), name='masses'),
        Input(shape=(None, 1), name='y_eqs'), ]
    query_u_input = Input(shape=(1, 1), name='input_query_u_freqs')
    query = Dense(mem_dim, use_bias=False)(query_u_input)


    # pp_denses = [Dense(mem_dim, use_bias=False)] * (n_protein_properties - 1)
    pp_masked = [Masking(mask_value=-1)(pp) for pp in protein_properties]
    pp_densed = [Dense(mem_dim, use_bias=False)(pp) for pp in pp_masked]
    pp_conv = [Conv1D(mem_dim, 6, padding='same',)(pp) for pp in pp_densed]
    #import tensorflow.keras.backend as K
    #print(K.int_shape(pp_conv), K.int_shape(pp_densed))
    pp_densed = [Add()([ppc, ppd]) for ppc, ppd in zip(pp_conv, pp_densed)]

    # memories = [MemoryRepresentation(output_dim=mem_dim, num_hops=1)]*(n_protein_properties-1)
    for _ in range(num_hops):
        outputs = [Attention()([query, pp]) for pp in pp_densed]
        query = Add()(outputs + [query])
        query = LayerNormalization()(query)

    answer = Dense(1)(query)
    answer = Activation('exponential')(answer)

    qa_model = Model(protein_properties + [query_u_input], answer)
    adam = tf.keras.optimizers.Adam(learning_rate=lr)
    qa_model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])

    return qa_model
