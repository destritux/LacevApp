import os
import numpy as np
import pandas as pd # Keep for type hinting or if DataFrames are directly processed
from scipy.signal import butter, filtfilt, iirnotch

# Assuming SignalPlotter will be in its own module later or passed as a dependency.
# from .signal_plotter import SignalPlotter # Or however it will be structured

class SignalFilter:
    def __init__(self, filtered_data_output_path: str,
                 high_pass_freq: float = 0.05,
                 low_pass_freq: float = 32.0, # Ensure float for consistency if used in division
                 filter_order: int = 4):

        if not filtered_data_output_path:
            raise ValueError("filtered_data_output_path cannot be empty or None.")

        self.filtered_data_output_path = filtered_data_output_path

        # Filter configuration
        self.config_lowcut_freq = np.float16(high_pass_freq)  # For bandpass, this is the lower bound
        self.config_highcut_freq = np.float16(low_pass_freq) # For bandpass, this is the upper bound
        self.config_filter_order = np.int8(filter_order)

    def apply_filters_and_save(self, raw_signal_data, output_subdir_name: str,
                               output_file_name_prefix: str, sample_rate: float = 250.0,
                               notch_frequency: float = 60.0, quality_factor: float = 10.0,
                               show_plot: bool = False):
        """
        Applies filtering to the data and saves the filtered signal.
        """
        if isinstance(raw_signal_data, (np.ndarray, np.generic)):
            signal_numpy_array = raw_signal_data
        elif isinstance(raw_signal_data, pd.Series):
            signal_numpy_array = np.float64(raw_signal_data.to_numpy().ravel())
        elif isinstance(raw_signal_data, pd.DataFrame):
            # Assuming data is in the first column if DataFrame
            signal_numpy_array = np.float64(raw_signal_data.iloc[:, 0].to_numpy().ravel())
        else:
            raise TypeError("raw_signal_data must be a NumPy array, Pandas Series, or Pandas DataFrame.")

        sampling_frequency_float = np.float64(sample_rate)  # Use float for sample_rate
        notch_freq_float = np.float64(notch_frequency)
        q_factor_float = np.float64(quality_factor)

        # Normalized frequency for notch filter (w0 = f0 / (fs / 2))
        normalized_notch_freq = notch_freq_float / (sampling_frequency_float / 2.0)

        notch_filtered_signal = self._apply_notch_filter(signal_numpy_array, normalized_notch_freq, q_factor_float)
        bandpass_filtered_signal = self._apply_butterworth_bandpass_filter(notch_filtered_signal, sampling_frequency_float)

        save_status_or_exception = self._save_signal_to_txt(bandpass_filtered_signal, output_file_name_prefix, output_subdir_name)

        if save_status_or_exception is True:
            if show_plot:
                # The SignalPlotter class is not available in this context yet,
                # or its instantiation might depend on paths not available here.
                # Commenting out plotting for now. This can be re-enabled once SignalPlotter is refactored
                # and properly integrated (e.g., by passing a SignalPlotter instance or necessary paths).
                # print("Plotting is currently disabled in SignalFilter.apply_filters_and_save due to refactoring.")
                # try:
                #     plot_utility_instance = SignalPlotter() # This would need path configurations
                #     # Ensure plot_raw_vs_filtered can access necessary paths if it uses them internally
                #     plot_utility_instance.plot_raw_vs_filtered(signal_numpy_array, bandpass_filtered_signal)
                # except NameError: # If SignalPlotter is not defined
                #     print("SignalPlotter class not found for plotting.")
                # except Exception as e:
                #     print(f"Error during plotting: {e}")
                pass

            # Return raw numpy, filtered numpy, and original (if it was a DataFrame/Series for context)
            return [np.array(signal_numpy_array), np.array(bandpass_filtered_signal), raw_signal_data]
        else:
            print(f'Error saving filtered signal: {save_status_or_exception}')
            return None # Indicate failure

    def _apply_notch_filter(self, input_signal: np.ndarray, normalized_center_freq: float, q_factor: float) -> np.ndarray:
        numerator_coeffs, denominator_coeffs = iirnotch(normalized_center_freq, q_factor)
        filtered_signal = filtfilt(numerator_coeffs, denominator_coeffs, input_signal)
        return np.array(filtered_signal, dtype=np.float64)

    def _apply_butterworth_bandpass_filter(self, input_signal: np.ndarray, sampling_frequency: float) -> np.ndarray:
        nyquist_frequency = 0.5 * sampling_frequency
        normalized_low_cutoff = self.config_lowcut_freq / nyquist_frequency
        normalized_high_cutoff = self.config_highcut_freq / nyquist_frequency

        # Ensure low < high for bandpass
        if normalized_low_cutoff >= normalized_high_cutoff:
            raise ValueError(f"Low cutoff frequency ({self.config_lowcut_freq} Hz) must be lower than high cutoff frequency ({self.config_highcut_freq} Hz) for bandpass filter.")

        numerator_coeffs, denominator_coeffs = butter(
            self.config_filter_order,
            [normalized_low_cutoff, normalized_high_cutoff],
            btype='bandpass'
        )
        filtered_signal = filtfilt(numerator_coeffs, denominator_coeffs, input_signal)
        return np.array(filtered_signal, dtype=np.float64)

    def _save_signal_to_txt(self, signal_to_save: np.ndarray, file_name_prefix: str, output_subdir_name: str):
        """Saves the signal to a .txt file."""
        # Construct the full path: base_output_path / subdir_name / file_prefix.txt
        class_specific_filtered_path = os.path.join(self.filtered_data_output_path, output_subdir_name)

        try:
            os.makedirs(class_specific_filtered_path, exist_ok=True)
            full_save_path = os.path.join(class_specific_filtered_path, file_name_prefix + '.txt')
            np.savetxt(full_save_path, signal_to_save, fmt='%1.9f')
            return True
        except Exception as er:
            # Return the exception to be handled or logged by the caller
            return er
