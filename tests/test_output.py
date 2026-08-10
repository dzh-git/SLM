import numpy as np

from slm_splitter.output import phase_to_uint8


def test_8bit_phase_export_preserves_all_codes():
    phase = (np.arange(256, dtype=np.float32) * (2.0 * np.pi / 256.0))[None, :]
    gray = phase_to_uint8(phase)

    assert np.array_equal(gray[0], np.arange(256, dtype=np.uint8))
