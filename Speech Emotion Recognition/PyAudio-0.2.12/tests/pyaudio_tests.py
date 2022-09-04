# -*- coding: utf-8 -*-
"""Automated unit tests for testing audio playback and capture.

These tests require an OS loopback sound device that forwards audio
output, generated by PyAudio for playback, and forwards it to an input
device, which PyAudio can record and verify against a test signal.

On Mac OS X, Soundflower can create such a device.

On GNU/Linux, the snd-aloop kernel module provides a loopback ALSA
device. Use examples/system_info.py to identify the name of the loopback
device.
"""

import math
import os
import struct
import time
import threading
import unittest
import wave
import sys

import numpy

import pyaudio

# To skip tests requiring hardware, set this environment variable:
SKIP_HW_TESTS = 'PYAUDIO_SKIP_HW_TESTS' in os.environ
# To run tests that require a loopback device (disabled by default), set this
# variable. If SKIP_HW_TESTS is set, this variable has no effect.
ENABLE_LOOPBACK_TESTS = 'PYAUDIO_ENABLE_LOOPBACK_TESTS' in os.environ

DUMP_CAPTURE=False

class PyAudioTests(unittest.TestCase):
    def setUp(self):
        self.p = pyaudio.PyAudio()
        self.loopback_input_idx = None
        self.loopback_output_idx = None

        if ENABLE_LOOPBACK_TESTS:
            (self.loopback_input_idx,
             self.loopback_output_idx) = self.get_audio_loopback()
            self.assertTrue(
                self.loopback_input_idx is None
                or self.loopback_input_idx >= 0, "No loopback device found")
            self.assertTrue(
                self.loopback_output_idx is None
                or self.loopback_output_idx >= 0, "No loopback device found")

    def tearDown(self):
        self.p.terminate()

    def get_audio_loopback(self):
        if sys.platform == 'darwin':
            return self._find_audio_loopback(
                'Soundflower (2ch)', 'Soundflower (2ch)')
        if sys.platform in ('linux', 'linux2'):
            return self._find_audio_loopback(
                'Loopback: PCM (hw:1,0)', 'Loopback: PCM (hw:1,1)')
        if sys.platform == 'win32':
            # Assumes running in a VM, in which the hypervisor can
            # set up a loopback device to back the "default" audio devices.
            # Here, None indicates default device.
            return None, None

        return -1, -1

    def _find_audio_loopback(self, indev, outdev):
        """Utility to find audio loopback device."""
        input_idx, output_idx = -1, -1
        for device_idx in range(self.p.get_device_count()):
            devinfo = self.p.get_device_info_by_index(device_idx)
            if (outdev == devinfo.get('name') and
                devinfo.get('maxOutputChannels', 0) > 0):
                output_idx = device_idx

            if (indev == devinfo.get('name') and
                devinfo.get('maxInputChannels', 0) > 0):
                input_idx = device_idx

            if output_idx > -1 and input_idx > -1:
                break

        return input_idx, output_idx

    @unittest.skipIf(SKIP_HW_TESTS, 'Loopback device required.')
    def test_system_info(self):
        """Basic system info tests"""
        self.assertTrue(self.p.get_host_api_count() > 0)
        self.assertTrue(self.p.get_device_count() > 0)
        api_info = self.p.get_host_api_info_by_index(0)
        self.assertTrue(len(api_info.items()) > 0)

    @unittest.skipIf(SKIP_HW_TESTS or not ENABLE_LOOPBACK_TESTS,
                     'Loopback device required.')
    def test_input_output_blocking(self):
        """Test blocking-based record and playback."""
        rate = 44100 # frames per second
        width = 2    # bytes per sample
        channels = 2
        # Blocking-mode might add some initial choppiness on some
        # platforms/loopback devices, so set a longer duration.
        duration = 3 # seconds
        frames_per_chunk = 1024

        freqs = [130.81, 329.63, 440.0, 466.16, 587.33, 739.99]
        test_signal = self.create_reference_signal(freqs, rate, width, duration)
        audio_chunks = self.signal_to_chunks(
            test_signal, frames_per_chunk, channels)

        out_stream = self.p.open(
            format=self.p.get_format_from_width(width),
            channels=channels,
            rate=rate,
            output=True,
            frames_per_buffer=frames_per_chunk,
            output_device_index=self.loopback_output_idx)
        in_stream = self.p.open(
            format=self.p.get_format_from_width(width),
            channels=channels,
            rate=rate,
            input=True,
            frames_per_buffer=frames_per_chunk,
            input_device_index=self.loopback_input_idx)

        captured = []
        for chunk in audio_chunks:
            out_stream.write(chunk)
            captured.append(in_stream.read(frames_per_chunk))
        # Capture a few more frames, since there is some lag.
        for i in range(8):
            captured.append(in_stream.read(frames_per_chunk))

        in_stream.stop_stream()
        out_stream.stop_stream()

        if DUMP_CAPTURE:
            self.write_wav('test_blocking.wav', b''.join(captured),
                           width, channels, rate)

        captured_signal = self.pcm16_to_numpy(b''.join(captured))
        captured_left_channel = captured_signal[::2]
        captured_right_channel = captured_signal[1::2]

        self.assert_pcm16_spectrum_nearly_equal(
            rate,
            captured_left_channel,
            test_signal,
            len(freqs))
        self.assert_pcm16_spectrum_nearly_equal(
            rate,
            captured_right_channel,
            test_signal,
            len(freqs))

    @unittest.skipIf(SKIP_HW_TESTS or not ENABLE_LOOPBACK_TESTS,
                     'Loopback device required.')
    def test_input_output_callback(self):
        """Test callback-based record and playback."""
        rate = 44100 # frames per second
        width = 2    # bytes per sample
        channels = 2
        duration = 1 # second
        frames_per_chunk = 1024

        freqs = [130.81, 329.63, 440.0, 466.16, 587.33, 739.99]
        test_signal = self.create_reference_signal(freqs, rate, width, duration)
        audio_chunks = self.signal_to_chunks(
            test_signal, frames_per_chunk, channels)

        state = {'count': 0}
        def out_callback(_, frame_count, time_info, status):
            if state['count'] >= len(audio_chunks):
                return ('', pyaudio.paComplete)
            rval = (audio_chunks[state['count']], pyaudio.paContinue)
            state['count'] += 1
            return rval

        captured = []
        def in_callback(in_data, frame_count, time_info, status):
            captured.append(in_data)
            return (None, pyaudio.paContinue)

        out_stream = self.p.open(
            format=self.p.get_format_from_width(width),
            channels=channels,
            rate=rate,
            output=True,
            frames_per_buffer=frames_per_chunk,
            output_device_index=self.loopback_output_idx,
            stream_callback=out_callback)

        in_stream = self.p.open(
            format=self.p.get_format_from_width(width),
            channels=channels,
            rate=rate,
            input=True,
            frames_per_buffer=frames_per_chunk,
            input_device_index=self.loopback_input_idx,
            stream_callback=in_callback)

        in_stream.start_stream()
        out_stream.start_stream()
        time.sleep(duration + 1)
        in_stream.stop_stream()
        out_stream.stop_stream()

        if DUMP_CAPTURE:
            self.write_wav('test_callback.wav', b''.join(captured),
                           width, channels, rate)

        captured_signal = self.pcm16_to_numpy(b''.join(captured))
        captured_left_channel = captured_signal[::2]
        captured_right_channel = captured_signal[1::2]

        self.assert_pcm16_spectrum_nearly_equal(
            rate,
            captured_left_channel,
            test_signal,
            len(freqs))
        self.assert_pcm16_spectrum_nearly_equal(
            rate,
            captured_right_channel,
            test_signal,
            len(freqs))

    @unittest.skipIf(SKIP_HW_TESTS, 'Loopback device required.')
    def test_device_lock_gil_order(self):
        """Ensure no deadlock between Pa_{Open,Start,Stop}Stream and GIL."""
        # This test targets macOS CoreAudio, which seems to use audio device
        # locks. The test is less relevant on ALSA and Win32 MME, which don't
        # seem to suffer even if the GIL is held while calling PortAudio.
        def in_callback(in_data, frame_count, time_info, status):
            # Release the GIL for a bit, but on macOS, still hold the device
            # lock.
            time.sleep(1)
            # Note: on macOS, must return paContinue; paComplete will deadlock
            # in the underlying call to AudioOutputUnitStop.
            return (None, pyaudio.paContinue)

        in_stream = self.p.open(
            format=self.p.get_format_from_width(2),
            channels=2,
            rate=44100,
            input=True,
            start=False,
            input_device_index=self.loopback_input_idx,
            stream_callback=in_callback)
        # In a separate (C) thread, portaudio/driver will grab the device lock,
        # then the GIL to call in_callback.
        in_stream.start_stream()
        # Wait a bit to let that callback thread start.
        time.sleep(0.5)
        # in_callback will eventually drop the GIL when executing time.sleep
        # (while retaining the device lock), allowing the following code to
        # run. All stream operations MUST release the GIL before attempting to
        # acquire the device lock. If that discipline is violated, the following
        # code would wait for the device lock while holding the GIL, while the
        # in_callback thread would be waiting for the GIL once time.sleep
        # completes (while holding the device lock), leading to deadlock.
        in_stream.stop_stream()

    @unittest.skipIf(SKIP_HW_TESTS, 'Loopback device required.')
    def test_stream_state_gil(self):
        """Ensure no deadlock between Pa_IsStream{Active,Stopped} and GIL."""
        rate = 44100 # frames per second
        width = 2    # bytes per sample
        channels = 2
        frames_per_chunk = 1024

        def out_callback(_, frame_count, time_info, status):
            return ('', pyaudio.paComplete)

        def in_callback(in_data, frame_count, time_info, status):
            # Release the GIL for a bit
            time.sleep(1)
            return (None, pyaudio.paComplete)

        in_stream = self.p.open(
            format=self.p.get_format_from_width(width),
            channels=channels,
            rate=rate,
            input=True,
            start=False,
            frames_per_buffer=frames_per_chunk,
            input_device_index=self.loopback_input_idx,
            stream_callback=in_callback)
        out_stream = self.p.open(
            format=self.p.get_format_from_width(width),
            channels=channels,
            rate=rate,
            output=True,
            start=False,
            frames_per_buffer=frames_per_chunk,
            output_device_index=self.loopback_output_idx,
            stream_callback=out_callback)
        # In a separate (C) thread, portaudio/driver will grab the device lock,
        # then the GIL to call in_callback.
        in_stream.start_stream()
        # Wait a bit to let that callback thread start.
        time.sleep(0.5)
        # in_callback will eventually drop the GIL when executing
        # time.sleep (while retaining the device lock), allowing the
        # following code to run. Checking the state of the stream MUST
        # not require the device lock, but if it does, it must release the GIL
        # before attempting to acquire the device
        # lock. Otherwise, the following code will wait for the device
        # lock (while holding the GIL), while the in_callback thread
        # will be waiting for the GIL once time.sleep completes (while
        # holding the device lock), leading to deadlock.
        self.assertTrue(in_stream.is_active())
        self.assertFalse(in_stream.is_stopped())

        self.assertTrue(out_stream.is_stopped())
        self.assertFalse(out_stream.is_active())
        out_stream.start_stream()
        self.assertFalse(out_stream.is_stopped())
        self.assertTrue(out_stream.is_active())
        time.sleep(1)
        in_stream.stop_stream()
        out_stream.stop_stream()

    @unittest.skipIf(SKIP_HW_TESTS, 'Loopback device required.')
    def test_get_stream_time_gil(self):
        """Ensure no deadlock between PA_GetStreamTime and GIL."""
        rate = 44100 # frames per second
        width = 2    # bytes per sample
        channels = 2
        frames_per_chunk = 1024

        def out_callback(_, frame_count, time_info, status):
            return ('', pyaudio.paComplete)

        def in_callback(in_data, frame_count, time_info, status):
            # Release the GIL for a bit
            time.sleep(1)
            return (None, pyaudio.paComplete)

        in_stream = self.p.open(
            format=self.p.get_format_from_width(width),
            channels=channels,
            rate=rate,
            input=True,
            start=False,
            frames_per_buffer=frames_per_chunk,
            input_device_index=self.loopback_input_idx,
            stream_callback=in_callback)
        out_stream = self.p.open(
            format=self.p.get_format_from_width(width),
            channels=channels,
            rate=rate,
            output=True,
            start=False,
            frames_per_buffer=frames_per_chunk,
            output_device_index=self.loopback_output_idx,
            stream_callback=out_callback)
        # In a separate (C) thread, portaudio/driver will grab the device lock,
        # then the GIL to call in_callback.
        in_stream.start_stream()
        # Wait a bit to let that callback thread start.
        time.sleep(0.5)
        # in_callback will eventually drop the GIL when executing
        # time.sleep (while retaining the device lock), allowing the
        # following code to run. Getting the stream time MUST not
        # require the device lock, but if it does, it must release the
        # GIL before attempting to acquire the device lock. Otherwise,
        # the following code will wait for the device lock (while
        # holding the GIL), while the in_callback thread will be
        # waiting for the GIL once time.sleep completes (while holding
        # the device lock), leading to deadlock.
        self.assertGreater(in_stream.get_time(), -1)
        self.assertGreater(out_stream.get_time(), 1)

        time.sleep(1)
        in_stream.stop_stream()
        out_stream.stop_stream()

    @unittest.skipIf(SKIP_HW_TESTS, 'Loopback device required.')
    def test_get_stream_cpuload_gil(self):
        """Ensure no deadlock between Pa_GetStreamCpuLoad and GIL."""
        rate = 44100 # frames per second
        width = 2    # bytes per sample
        channels = 2
        frames_per_chunk = 1024

        def out_callback(_, frame_count, time_info, status):
            return ('', pyaudio.paComplete)

        def in_callback(in_data, frame_count, time_info, status):
            # Release the GIL for a bit
            time.sleep(1)
            return (None, pyaudio.paComplete)

        in_stream = self.p.open(
            format=self.p.get_format_from_width(width),
            channels=channels,
            rate=rate,
            input=True,
            start=False,
            frames_per_buffer=frames_per_chunk,
            input_device_index=self.loopback_input_idx,
            stream_callback=in_callback)
        out_stream = self.p.open(
            format=self.p.get_format_from_width(width),
            channels=channels,
            rate=rate,
            output=True,
            start=False,
            frames_per_buffer=frames_per_chunk,
            output_device_index=self.loopback_output_idx,
            stream_callback=out_callback)
        # In a separate (C) thread, portaudio/driver will grab the device lock,
        # then the GIL to call in_callback.
        in_stream.start_stream()
        # Wait a bit to let that callback thread start.
        time.sleep(0.5)
        # in_callback will eventually drop the GIL when executing
        # time.sleep (while retaining the device lock), allowing the
        # following code to run. Getting the stream cpuload MUST not
        # require the device lock, but if it does, it must release the
        # GIL before attempting to acquire the device lock. Otherwise,
        # the following code will wait for the device lock (while
        # holding the GIL), while the in_callback thread will be
        # waiting for the GIL once time.sleep completes (while holding
        # the device lock), leading to deadlock.
        self.assertGreater(in_stream.get_cpu_load(), -1)
        self.assertGreater(out_stream.get_cpu_load(), -1)
        time.sleep(1)
        in_stream.stop_stream()
        out_stream.stop_stream()

    @unittest.skipIf(SKIP_HW_TESTS, 'Loopback device required.')
    def test_get_stream_write_available_gil(self):
        """Ensure no deadlock between Pa_GetStreamWriteAvailable and GIL."""
        rate = 44100 # frames per second
        width = 2    # bytes per sample
        channels = 2
        frames_per_chunk = 1024

        def in_callback(in_data, frame_count, time_info, status):
            # Release the GIL for a bit
            time.sleep(1)
            return (None, pyaudio.paComplete)

        in_stream = self.p.open(
            format=self.p.get_format_from_width(width),
            channels=channels,
            rate=rate,
            input=True,
            start=False,
            frames_per_buffer=frames_per_chunk,
            input_device_index=self.loopback_input_idx,
            stream_callback=in_callback)
        out_stream = self.p.open(
            format=self.p.get_format_from_width(width),
            channels=channels,
            rate=rate,
            output=True,
            frames_per_buffer=frames_per_chunk,
            output_device_index=self.loopback_output_idx)
        # In a separate (C) thread, portaudio/driver will grab the device lock,
        # then the GIL to call in_callback.
        in_stream.start_stream()
        # Wait a bit to let that callback thread start.
        time.sleep(0.5)
        # in_callback will eventually drop the GIL when executing
        # time.sleep (while retaining the device lock), allowing the
        # following code to run. Getting the stream write available MUST not
        # require the device lock, but if it does, it must release the
        # GIL before attempting to acquire the device lock. Otherwise,
        # the following code will wait for the device lock (while
        # holding the GIL), while the in_callback thread will be
        # waiting for the GIL once time.sleep completes (while holding
        # the device lock), leading to deadlock.
        self.assertGreater(out_stream.get_write_available(), -1)

        time.sleep(1)
        in_stream.stop_stream()

    @unittest.skipIf(SKIP_HW_TESTS, 'Loopback device required.')
    def test_get_stream_read_available_gil(self):
        """Ensure no deadlock between Pa_GetStreamReadAvailable and GIL."""
        rate = 44100 # frames per second
        width = 2    # bytes per sample
        channels = 2
        frames_per_chunk = 1024

        def out_callback(in_data, frame_count, time_info, status):
            # Release the GIL for a bit
            time.sleep(1)
            return (None, pyaudio.paComplete)

        in_stream = self.p.open(
            format=self.p.get_format_from_width(width),
            channels=channels,
            rate=rate,
            input=True,
            frames_per_buffer=frames_per_chunk,
            input_device_index=self.loopback_input_idx)
        out_stream = self.p.open(
            format=self.p.get_format_from_width(width),
            channels=channels,
            rate=rate,
            output=True,
            start=False,
            frames_per_buffer=frames_per_chunk,
            output_device_index=self.loopback_output_idx,
            stream_callback=out_callback)
        # In a separate (C) thread, portaudio/driver will grab the device lock,
        # then the GIL to call in_callback.
        out_stream.start_stream()
        # Wait a bit to let that callback thread start.
        time.sleep(0.5)
        # in_callback will eventually drop the GIL when executing
        # time.sleep (while retaining the device lock), allowing the
        # following code to run. Getting the stream read available MUST not
        # require the device lock, but if it does, it must release the
        # GIL before attempting to acquire the device lock. Otherwise,
        # the following code will wait for the device lock (while
        # holding the GIL), while the in_callback thread will be
        # waiting for the GIL once time.sleep completes (while holding
        # the device lock), leading to deadlock.
        self.assertGreater(in_stream.get_read_available(), -1)

        time.sleep(1)
        in_stream.stop_stream()

    @unittest.skipIf(SKIP_HW_TESTS, 'Loopback device required.')
    def test_terminate_gil(self):
        """Ensure no deadlock between Pa_Terminate and GIL."""
        # This test targets macOS CoreAudio, which seems to use audio device
        # locks. The test is less relevant on ALSA and Win32 MME, which don't
        # seem to suffer even if the GIL is held while calling PortAudio.
        width = 2
        channels = 2
        bytes_per_frame = width * channels
        event = threading.Event()

        def out_callback(in_data, frame_count, time_info, status):
            event.set()
            time.sleep(0.5)  # Release the GIL for a bit
            event.clear()
            return (b'\1' * frame_count * bytes_per_frame, pyaudio.paContinue)

        out_stream = self.p.open(
            format=self.p.get_format_from_width(width),
            channels=channels,
            rate=44100,
            output=True,
            start=False,
            output_device_index=self.loopback_output_idx,
            stream_callback=out_callback)
        # In a separate (C) thread, portaudio/driver will grab the device lock,
        # then the GIL to call in_callback.
        out_stream.start_stream()
        # Wait a bit to let that callback thread start. For output streams on
        # macOS, it's important to have one complete call to out_callback before
        # attempting to terminate. Otherwise, the underlying call to
        # AudioOutputUnitStop will deadlock.
        time.sleep(0.6)
        # out_callback will eventually drop the GIL when executing time.sleep
        # (while retaining the device lock), allowing the following code to
        # run. Terminating PyAudio MUST release the GIL before attempting to
        # acquire the device lock (if the lock is needed). If that discipline is
        # violated, the following code would wait for the device lock while
        # holding the GIL, while the out_callback thread would be waiting for
        # the GIL once time.sleep completes (while holding the device lock),
        # leading to deadlock.
        #
        # Wait until out_callback is about to sleep, plus a little extra to
        # help ensure that sleep() is called first before we concurrently call
        # into self.p.terminate().
        event.wait()
        time.sleep(0.1)
        self.p.terminate()

    @unittest.skipIf(SKIP_HW_TESTS, 'Loopback device required.')
    def test_return_none_callback(self):
        """Ensure that return None ends the stream."""
        num_times_called = 0

        def out_callback(_, frame_count, time_info, status):
            nonlocal num_times_called
            num_times_called += 1
            return (None, pyaudio.paContinue)

        out_stream = self.p.open(
            format=self.p.get_format_from_width(2),
            channels=2,
            rate=44100,
            output=True,
            output_device_index=self.loopback_output_idx,
            stream_callback=out_callback)
        out_stream.start_stream()
        time.sleep(0.5)
        out_stream.stop_stream()
        self.assertEqual(num_times_called, 1)

    @unittest.skipIf(SKIP_HW_TESTS, 'Loopback device required.')
    def test_excess_output_callback(self):
        """Ensure that returning more bytes than allowed does not fail."""
        num_times_called = 0
        width = 2
        channels = 2
        bytes_per_frame = width * channels

        def out_callback(_, frame_count, time_info, status):
            nonlocal num_times_called
            num_times_called += 1
            # Make sure this is called twice, so we know that the first time
            # didn't crash (at least).
            result = (pyaudio.paComplete
                      if num_times_called == 2 else pyaudio.paContinue)
            max_allowed_bytes = frame_count * bytes_per_frame
            return (b'\1' * (max_allowed_bytes * 2), result)

        out_stream = self.p.open(
            format=self.p.get_format_from_width(width),
            channels=channels,
            rate=44100,
            output=True,
            output_device_index=self.loopback_output_idx,
            stream_callback=out_callback)
        out_stream.start_stream()
        time.sleep(0.5)
        out_stream.stop_stream()
        self.assertEqual(num_times_called, 2)

    @staticmethod
    def create_reference_signal(freqs, sampling_rate, width, duration):
        """Return reference signal with several sinuoids with frequencies
        specified by freqs."""
        total_frames = int(sampling_rate * duration)
        max_amp = float(2**(width * 8 - 1) - 1)
        avg_amp = max_amp / len(freqs)
        return [
            int(sum(avg_amp * math.sin(2*math.pi*freq*(k/float(sampling_rate)))
                for freq in freqs))
            for k in range(total_frames)]

    @staticmethod
    def signal_to_chunks(frame_data, frames_per_chunk, channels):
        """Given an array of values comprising the signal, return an iterable
        of binary chunks, with each chunk containing frames_per_chunk
        frames. Each frame represents a single value from the signal,
        duplicated for each channel specified by channels.
        """
        frames = [struct.pack('h', x) * channels for x in frame_data]
        # Chop up frames into chunks
        return [b''.join(chunk_frames) for chunk_frames in
                tuple(frames[i:i+frames_per_chunk]
                      for i in range(0, len(frames), frames_per_chunk))]

    @staticmethod
    def pcm16_to_numpy(bytestring):
        """From PCM 16-bit bytes, return an equivalent numpy array of values."""
        return struct.unpack('%dh' % (len(bytestring) / 2), bytestring)

    @staticmethod
    def write_wav(filename, data, width, channels, rate):
        """Write PCM data to wave file."""
        wf = wave.open(filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(width)
        wf.setframerate(rate)
        wf.writeframes(data)
        wf.close()

    def assert_pcm16_spectrum_nearly_equal(self, sampling_rate, cap, ref,
                                           num_freq_peaks_expected):
        """Compares the discrete fourier transform of a captured signal
        against the reference signal and ensures that the frequency peaks
        match."""
        # When passing a reference signal through the loopback device,
        # the captured signal may include additional noise, as well as
        # time lag, so testing that the captured signal is "similar
        # enough" to the reference using bit-wise equality won't work
        # well. Instead, the approach here a) assumes the reference
        # signal is a sum of sinusoids and b) computes the discrete
        # fourier transform of the reference and captured signals, and
        # ensures that the frequencies of the top
        # num_freq_peaks_expected frequency peaks are close.
        cap_fft = numpy.absolute(numpy.fft.rfft(cap))
        ref_fft = numpy.absolute(numpy.fft.rfft(ref))
        # Find the indices of the peaks:
        cap_peak_indices = sorted(numpy.argpartition(
            cap_fft, -num_freq_peaks_expected)[-num_freq_peaks_expected:])
        ref_peak_indices = sorted(numpy.argpartition(
            ref_fft, -num_freq_peaks_expected)[-num_freq_peaks_expected:])
        # Ensure that the corresponding frequencies of the peaks are close:
        for cap_freq_index, ref_freq_index in zip(cap_peak_indices,
                                                  ref_peak_indices):
            cap_freq = cap_freq_index / float(len(cap)) * (sampling_rate / 2)
            ref_freq = ref_freq_index / float(len(ref)) * (sampling_rate / 2)
            diff = abs(cap_freq - ref_freq)
            self.assertLess(diff, 1.0)

        # As an additional test, verify that the spectrum (not just
        # the peaks) of the reference and captured signal are similar
        # by computing the cross-correlation of the spectra. Assuming they
        # are nearly identical, the cross-correlation should contain a large
        # peak when the spectra overlap and mostly 0s elsewhere. Verify that
        # using a histogram of the cross-correlation:
        freq_corr_hist, _ = numpy.histogram(
            numpy.correlate(cap_fft, ref_fft, mode='full'),
            bins=10)
        self.assertLess(sum(freq_corr_hist[2:])/sum(freq_corr_hist), 1e-2)