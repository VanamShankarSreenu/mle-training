
import importlib.util


def test_installation():
    import mle_training
    import mle_training.ingest_data
    import mle_training.score
    import mle_training.train
    package_name = 'mle_training'
    assert importlib.util.find_spec(package_name) is not None


test_installation()