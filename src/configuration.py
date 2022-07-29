class Config:
    def __init__(self,
                 train_set: str = "train",
                 test_set: str = "devel",
                 acoustic_llds: str = "",
                 acoustic_functionals: str = "",
                 acoustic_pca=0,
                 acoustic_gmm=0,
                 acoustic_pfi=0,
                 bert_model: str = "",
                 linguistic_llds: str = "",
                 linguistic_functionals: str = "",
                 linguistic_pca=0,
                 linguistic_gmm=0,
                 linguistic_pfi=0,
                 overwrite_pca: bool = False,
                 overwrite_gmm: bool = False,
                 overwrite_pfi: bool = False,
                 elm_c=[1],
                 power_norm_gamma=[1],
                 old_order: bool = False):

        self.train_set = train_set
        self.test_set = test_set

        self.acoustic_llds = acoustic_llds
        self.acoustic_functionals = acoustic_functionals
        self.acoustic_pca = acoustic_pca
        self.acoustic_gmm = acoustic_gmm
        self.acoustic_pfi = acoustic_pfi

        self.bert_model = bert_model
        self.linguistic_llds = linguistic_llds
        self.linguistic_functionals = linguistic_functionals
        self.linguistic_pca = linguistic_pca
        self.linguistic_gmm = linguistic_gmm
        self.linguistic_pfi = linguistic_pfi

        self.overwrite_pca = overwrite_pca
        self.overwrite_gmm = overwrite_gmm
        self.overwrite_pfi = overwrite_pfi

        self.elm_c = elm_c
        self.power_norm_gamma = power_norm_gamma

        self.old_order = old_order
