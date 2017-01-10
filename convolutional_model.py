

class ConvolutionalModel(Model):
    def __init__(self, conv_sizes, filter_sizes, pool_size, fc_sizes, input_shape=None, do_normalization=False, load_from=None):
        conv_init_arg = (input_shape[0], conv_sizes, filter_sizes, pool_size, load_from)
        fc_init_arg = (input_shape[0], fc_sizes, do_normalization, load_from)

        self.n_batch = input_shape[0]

        self.conv_part = ConvolutionPart(*conv_init_arg)
        self.fc_part = FCPart(*fc_init_arg)
        self.input_shape = input_shape
        self.input_shape[0] = None

    def add_to_graph(self, input_tensor):
        conv_out = self.conv_part.add_to_graph(input_tensor)
        q_out = self.fc_part.add_to_graph(self.conv_out)
        return q_out

    def add_assign_weights(self, key, rhs):
        self.conv_part.add_assign_weights(key, rhs.conv_part)
        self.fc_part.add_assign_weights(key, rhs.fc_part)

    def run_assign_weights(self, key, sess):
        self.conv_part.run_assign_weights(key, sess)
        self.fc_part.run_assign_weights(key, sess)

    def save_weights(self, file_name, sess):
        f = file_name
        if type(file_name) == str:
            f = open(file_name, 'wb')
        d1 = self.conv_part.saveable_weights_dict(f, sess)
        d2 = self.fc_part.saveable_weights_dict(f, sess)
        np.savez_compressed(f, **{**d1, **d2})
