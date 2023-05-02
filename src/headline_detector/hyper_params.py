hyper_params_indobertweet = {
    'model_name': "indolem/indobertweet-base-uncased",
    'seq_length': 256,
    'out_feature': 2,
    'learning_rate': 2e-5,
    'batch_size': 16
}

hyper_params_cnn = {
    'seq_length': 256,
    'out_feature': 2,
    'learning_rate': 1e-3,
    'batch_size': 16,
    'conv_num_filters': 100,
    'conv_kernels': (1,2,3)
}

hyper_params_fasttext = {
    'seq_length': 256,
    'out_feature': 2,
    'learning_rate': 8e-5,
    'batch_size': 64
}