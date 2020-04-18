from tensorflow.keras.layers import *
from tensorflow.keras.models import *

from covid19_resonances.neural_models.MemoryNetworkModel import MemoryRepresentation, OutputLayerW


def MemN2N_model(mem_dim=16, num_hops=3, n_protein_properties=5):
    # n_protein_properties: ks, charges, masses, y_eqs, freqs

    protein_properties = [Input(shape=(None, 1), name='input_{}'.format(i)) for i in range(n_protein_properties-1)]
    query_u_input = Input(shape=(1, 1), name='input_query_u_freqs')
    query = Dense(mem_dim, use_bias=False)(query_u_input)

    pp_denses = [Dense(mem_dim, use_bias=False)]*(n_protein_properties-1)
    pp_densed = [dense(pp) for dense, pp in zip(pp_denses, protein_properties)]

    #memories = [MemoryRepresentation(output_dim=mem_dim, num_hops=1)]*(n_protein_properties-1)
    for _ in range(num_hops):
        outputs = [Attention()([pp, query]) for pp in pp_densed]
        query = Add()(outputs + [query])

    answer = Dense(1, use_bias=False)(query)
    answer = Activation('exponential')(answer)

    qa_model = Model(protein_properties + [query_u_input], answer)
    qa_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return qa_model