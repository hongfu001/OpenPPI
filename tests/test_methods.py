"""Method registry: each recipe's components fit together and sampling is never shared."""
import pytest

from openppi import METHODS, InferenceMethod, SamplingDesign, get_method, multi_allocate


def test_recipes_fit_their_components():
    assert set(METHODS) == {'PPI', 'PPI++', 'RePPI', 'FAB-PPI', 'Cluster-PPI', 'MultiPPI', 'ASI', 'RASI',
                            'FTRL', 'BAI'}
    for name, recipe in METHODS.items():
        assert recipe.name == name and get_method(name) is recipe
        assert isinstance(recipe.inference, InferenceMethod)
        assert issubclass(recipe.data_type, recipe.inference.data_type)
        assert recipe.reference


def test_sampling_components_are_built_fresh():
    for name in ('ASI', 'FTRL', 'BAI'):
        first, second = get_method(name).sampling(), get_method(name).sampling()
        assert isinstance(first, SamplingDesign) and first is not second
    assert get_method('RASI').sampling.fit  # a class, fitted on burn-in labels
    assert get_method('MultiPPI').sampling is multi_allocate
    assert get_method('PPI').sampling is None and get_method('PPI++').inference.lam == 'tuned'
    with pytest.raises(KeyError, match='choose from'):
        get_method('PPI+++')
