from distutils.core import setup

setup(
    name="RNN-VD",
    version="0.1",
    description="""RNN with Gaussian Variational Dropout.""",
    packages=["rnn_vd"],
    requires=["cplxmodule", "pytorch"]
)
