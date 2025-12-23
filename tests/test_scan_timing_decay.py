import numpy as np
import pytest
from petpal.utils.scan_timing import calculate_frame_reference_time
from petpal.preproc.decay_correction import calculate_frame_decay_factor
from petpal.utils.scan_timing import ScanTimingInfo

def test_from_start_end_computes_duration_center_and_default_decay():
    starts = np.array([0.0, 60.0, 180.0])
    ends = np.array([60.0, 180.0, 360.0])
    info = ScanTimingInfo.from_start_end(frame_starts=starts, frame_ends=ends)
    np.testing.assert_allclose(info.duration, np.array([60.0, 120.0, 180.0]))
    np.testing.assert_allclose(info.center, np.array([30.0, 120.0, 270.0]))
    np.testing.assert_allclose(info.decay, np.ones_like(starts))

def test_from_start_end_uses_provided_decay_list():
    starts = np.array([0.0, 50.0])
    ends = np.array([25.0, 100.0])
    decay_list = [1.0, 0.9]
    info = ScanTimingInfo.from_start_end(frame_starts=starts, frame_ends=ends, decay_correction_factor=decay_list)
    np.testing.assert_allclose(info.duration, np.array([25.0, 50.0]))
    np.testing.assert_allclose(info.center, np.array([12.5, 75.0]))
    np.testing.assert_allclose(info.decay, np.array(decay_list))

def test_ref_time_no_decay_returns_midpoint():
    durations = np.array([5.0, 10.0])
    starts = np.array([0.0, 5.0])
    half_life = 1.0e8  # effectively no decay
    res = calculate_frame_reference_time(durations, starts, half_life)
    expected = starts + durations / 2.0
    np.testing.assert_allclose(res, expected, rtol=1e-2, atol=1e-3)


def test_ref_time_fast_decay_concentrates_near_start():
    durations = np.array([60.0, 30.0, 15.0])
    starts = np.array([0.0, 5.0, 10.0])
    half_life = 1e-6  # very fast decay
    res = calculate_frame_reference_time(durations, starts, half_life)
    delays = res - starts
    # For very fast decay, the reference time should be very close to frame start
    assert np.all(delays < durations * 0.01)


def test_ref_time_numeric_integration_agrees():
    # compare against numeric integral definition of weighted average time
    durations = np.array([5.0, 10.0, 60.0])
    starts = np.array([0.0, 5.0, 15.0])
    half_life = 1e4
    res = calculate_frame_reference_time(durations, starts, half_life)

    expected = []
    for T, s in zip(durations, starts):
        lam = np.log(2) / half_life
        t = np.linspace(0.0, T, 20001)
        w = np.exp(-lam * t)
        num = np.trapezoid(t * w, t)
        den = np.trapezoid(w, t)
        delay = num / den
        expected.append(s + delay)
    expected = np.asarray(expected)

    np.testing.assert_allclose(res, expected, rtol=1e-3, atol=1e-9)


def test_ref_time_vectorized_shape_and_broadcast():
    durations = np.array([10.0])
    starts = np.array([2.0])
    half_life = 1.0e3
    res = calculate_frame_reference_time(durations, starts, half_life)
    assert isinstance(res, np.ndarray)
    assert res.shape == (1,)


def test_basic_powers_of_two():
    half_life = 2.0
    times = np.array([0.0, half_life, 2 * half_life])
    out = calculate_frame_decay_factor(times, half_life)
    expected = np.array([1.0, 2.0, 4.0])
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=0)


def test_preserves_shape_and_dtype():
    times = np.array([0.5])
    out = calculate_frame_decay_factor(times, 1.0)
    assert isinstance(out, np.ndarray)
    assert out.shape == times.shape
    np.testing.assert_allclose(out[0], 2 ** 0.5, rtol=1e-12)


def test_negative_time_and_float_half_life():
    half_life = 1.5
    times = np.array([-half_life, 0.0, half_life])
    out = calculate_frame_decay_factor(times, half_life)
    expected = np.array([0.5, 1.0, 2.0])
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=0)
