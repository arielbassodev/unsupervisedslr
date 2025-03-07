class Shear(object):
    def __init__(self, s1=None, s2=None):
        self.s1 = s1
        self.s2 = s2

    def call(self, data_numpy):
        temp = data_numpy.copy()

        if self.s1 is not None:
            s1_list = self._extract_values(self.s1)
        else:
            s1_list = [random.uniform(-1, 1)]

        if self.s2 is not None:
            s2_list = self._extract_values(self.s2)
        else:
            s2_list = [random.uniform(-1, 1)]
        s1_list = s1_list[:1]  
        s2_list = s2_list[:1]
        R = np.array([[1, s1_list[0]],
                      [s2_list[0], 1]])

        temp = np.dot(temp, R)
        return temp

    def _extract_values(self, tensor_or_list):

        if isinstance(tensor_or_list, torch.Tensor):
            values = tensor_or_list.detach().cpu().numpy().tolist()
        else:
            values = list(tensor_or_list)
        if isinstance(values[0], (list, np.ndarray)):
            values = [v for sublist in values for v in sublist]

        return values

