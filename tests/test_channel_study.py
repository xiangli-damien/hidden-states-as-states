import numpy as np
import zarr
from hss.analysis.channel_data import token_samples


def test_token_alignment_and_short_response_missingness():
    values = np.arange(9 * 3, dtype=np.float32).reshape(9, 3)
    a = zarr.array(values, chunks=(4, 2))
    out = token_samples(a, np.array([0, 2, 9]), [1, 2, 4])
    np.testing.assert_array_equal(out[0, 0], values[0])
    np.testing.assert_array_equal(out[1, 0], values[2])
    np.testing.assert_array_equal(out[:, -1], values[[1, 8]])
    assert np.isnan(out[0, 2]).all()
    np.testing.assert_array_equal(out[1, 2], values[5])
