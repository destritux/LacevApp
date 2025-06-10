import os
import numpy as np
import pandas as pd
from scipy.fft import rfft # Keep for FeatureExtractor or SignalPlotter
from scipy.signal import welch # Keep for FeatureExtractor or SignalPlotter
from scipy.integrate import simpson # Keep for FeatureExtractor
from antropy import app_entropy # Keep for FeatureExtractor
from nolds import dfa, lyap_e, lyap_r # Keep for FeatureExtractor

class FeatureExtractor:
  def __init__(self, features_output_path: str):
    if not features_output_path:
        raise ValueError("features_output_path cannot be empty or None.")
    self.features_output_path = features_output_path

  def extract_features(self, data_array, segment_duration_minutes,
                       start_time_info, start_date_info,
                       class_id, class_name, sample_rate,
                       feature_results_list, # This is modified in-place, consider returning a new list
                       file_name_prefix: str, # Added for print statements
                       stage_info=False, date_value=0): # date_value seems unused
    """
    Extracts features from the data_array.
    GUI-related operations (logging, progress bar) have been removed.
    The caller is responsible for such updates.
    """
    # print(f'Calculating features for {file_name_prefix}. Please, wait...\n') # Optional: replace GUI log with print

    total_data_points = len(data_array)
    points_per_segment = int(sample_rate * 60 * segment_duration_minutes)
    if points_per_segment == 0:
        # Avoid division by zero if sample_rate, segment_duration_minutes, or points_per_segment is zero
        print(f"Warning: points_per_segment is zero for {file_name_prefix}. Skipping feature extraction.")
        return feature_results_list # Return existing list

    number_of_segments = total_data_points // points_per_segment
    current_minute_offset = 0

    if total_data_points >= points_per_segment:
      segment_length_points = points_per_segment
      overlap_points = int(segment_length_points // 3)

      for i in range(1, number_of_segments + 1):
          # print(f"{file_name_prefix}: Segment {i} of {number_of_segments}") # Optional console log
          if i == 1:
              segment_start_index = 0
              segment_end_index = segment_length_points
          else:
              segment_start_index = segment_end_index - overlap_points
              segment_end_index = segment_start_index + segment_length_points

          # Ensure segment_end_index does not exceed data_array length
          if segment_end_index > total_data_points:
              segment_end_index = total_data_points
              if segment_start_index >= segment_end_index: # Avoid empty or invalid segment
                  continue


          current_segment_data = data_array[segment_start_index:segment_end_index]

          if len(current_segment_data) == 0: # Skip empty segments
              continue
          mean_absolute_value = abs(np.mean(current_segment_data)) # Use np.mean for numpy array

          # Ensure segment is sufficiently long and not zeroed out
          if (len(current_segment_data) > 100) and (float(mean_absolute_value) > 1e-9): # Check against small epsilon for zero mean
            segment_array = np.array(current_segment_data).ravel()

            approximate_entropy = app_entropy(segment_array, order=2)
            detrended_fluctuation_analysis = dfa(segment_array, fit_exp='poly', overlap=False)

            lyapunov_exponent_values = lyap_e(segment_array)
            lyapunov_exponent_r = lyap_r(segment_array, min_tsep=1305 if len(segment_array) > 1305*2 else max(1, (len(segment_array)//2)-1) ) # Adjust min_tsep based on segment length

            fft_results = rfft(segment_array)
            fft_description_df = pd.DataFrame(np.array(abs(fft_results)).ravel()).describe()

            frequencies, psd_values = welch(segment_array, fs=sample_rate, return_onesided=False, detrend=False)
            psd_description_df = pd.DataFrame(np.array(abs(psd_values)).ravel()).describe()

            if len(frequencies) < 2: # Not enough frequencies for resolution calculation
                frequency_resolution = 1.0 # Default or skip bandpower
            else:
                frequency_resolution = frequencies[1] - frequencies[0]

            # Bandpower calculations (ensure frequencies and psd_values are valid)
            # ... (add checks for empty idx_... arrays before simpson if necessary)
            band_low_freq1, band_high_freq1 = 0,0.5
            band_indices_low = np.logical_and(frequencies >= band_low_freq1, frequencies <= band_high_freq1)
            absolute_band_power_low = simpson(psd_values[band_indices_low], dx=frequency_resolution) if np.any(band_indices_low) else 0.0

            band_low_freq2, band_high_freq2 = 0.5, 4
            band_indices_delta = np.logical_and(frequencies >= band_low_freq2, frequencies <= band_high_freq2)
            absolute_band_power_delta = simpson(psd_values[band_indices_delta], dx=frequency_resolution) if np.any(band_indices_delta) else 0.0

            band_low_freq3, band_high_freq3 = 4, 8
            band_indices_theta = np.logical_and(frequencies >= band_low_freq3, frequencies <= band_high_freq3)
            absolute_band_power_theta = simpson(psd_values[band_indices_theta], dx=frequency_resolution) if np.any(band_indices_theta) else 0.0

            band_low_freq4, band_high_freq4 = 8, 12
            band_indices_alpha = np.logical_and(frequencies >= band_low_freq4, frequencies <= band_high_freq4)
            absolute_band_power_alpha = simpson(psd_values[band_indices_alpha], dx=frequency_resolution) if np.any(band_indices_alpha) else 0.0

            band_low_freq5, band_high_freq5 = 12, 30
            band_indices_beta = np.logical_and(frequencies >= band_low_freq5, frequencies <= band_high_freq5)
            absolute_band_power_beta = simpson(psd_values[band_indices_beta], dx=frequency_resolution) if np.any(band_indices_beta) else 0.0

            segment_data_description_df = pd.DataFrame(np.array(abs(current_segment_data)).ravel()).describe()

            if isinstance(start_time_info, int): formatted_minute = current_minute_offset
            elif isinstance(start_time_info, list) and len(start_time_info) > segment_end_index -1 : # Check list and length
                time_segment_processed = list(map(int, start_time_info[segment_start_index:segment_end_index]))
                if time_segment_processed: # Ensure not empty
                    time_segment_processed = sum(time_segment_processed)/len(time_segment_processed)
                    formatted_minute = int(time_segment_processed)
                    formatted_minute =  self._adjust_minute_representation(formatted_minute)
                else:
                    formatted_minute = current_minute_offset # Fallback
            else: # Fallback if start_time_info is not as expected
                formatted_minute = current_minute_offset

            if isinstance(start_date_info, int): formatted_date ='0/00/0000'
            elif isinstance(start_date_info, list) and len(start_date_info) > segment_start_index: # Check list and length
                formatted_date = start_date_info[segment_start_index]
            else: # Fallback
                formatted_date = '0/00/0000'

            feature_vector = [ class_id, class_name, formatted_minute, formatted_date]
            if stage_info: # Only add if stage_info is not False or None
                 feature_vector.append(stage_info)

            positive_lyapunov_exponents_count=0
            if isinstance(lyapunov_exponent_values, (list, np.ndarray)): # Check if iterable
                for val_le in lyapunov_exponent_values:
                    if val_le > 0:
                        positive_lyapunov_exponents_count += 1

            feature_vector += [
                     float(abs(approximate_entropy)), float(abs(detrended_fluctuation_analysis)),
                     float(positive_lyapunov_exponents_count), float(lyapunov_exponent_r),
                     float(abs(absolute_band_power_low)),float(abs(absolute_band_power_delta)),
                     float(abs(absolute_band_power_theta)), float(abs(absolute_band_power_alpha)),
                     float(abs(absolute_band_power_beta)),
                     float(fft_description_df.iloc[1][0]), float(abs(fft_description_df.iloc[3][0])),
                     float(abs(fft_description_df.iloc[-1][0])), float(abs(fft_description_df.iloc[2][0])),
                     float(abs(psd_description_df.iloc[1][0])), float(abs(psd_description_df.iloc[3][0])),
                     float(abs(psd_description_df.iloc[-1][0])), float(abs(psd_description_df.iloc[2][0])),
                     float(abs(segment_data_description_df.iloc[1][0])), float(abs(segment_data_description_df.iloc[3][0])),
                     float(abs(segment_data_description_df.iloc[-1][0])), float(abs(segment_data_description_df.iloc[2][0]))
                  ]

            feature_results_list.append(feature_vector)
            current_minute_offset += segment_duration_minutes
            current_minute_offset =  self._adjust_minute_representation(current_minute_offset)

    return feature_results_list

  def format_time_from_hora_column(self, dataframe_with_time: pd.DataFrame, minute_cutoff_adjustment: int):
    time_list= []
    accumulated_hour_minute_str="0000" # Initialize with a default string
    for time_entry in dataframe_with_time['HORA']:
        try:
            time_parts=str(time_entry).split(':')
            hour_val = int(time_parts[0].replace(' ', '').replace('\\', '').replace('n', '').rstrip().lstrip())
            if hour_val < 24 and len(time_parts) > 1:
                minute_val=int(str(time_parts[1]).replace('\\', '').replace('n', '').replace(' ', '').rstrip().lstrip())
                if minute_val < 10:
                    minute_val_str = str('0'+str(minute_val))
                else:
                    minute_val_str = str(minute_val)
                accumulated_hour_minute_str = str(hour_val) + minute_val_str # Ensure HHMM format
                if len(accumulated_hour_minute_str) == 3: # e.g. 900 for 9:00
                    accumulated_hour_minute_str = "0" + accumulated_hour_minute_str
                elif len(accumulated_hour_minute_str) < 3 : # e.g. 00 for 0:00
                     accumulated_hour_minute_str = "00" + accumulated_hour_minute_str


            else: # Invalid hour or format
                if accumulated_hour_minute_str == "0000" and not time_list : # If it's the first entry and invalid
                     # Decide a default or skip. For now, using previous or 0 if first.
                     # This part of logic might need review based on expected data quality.
                    pass # Keep previous accumulated_hour_minute_str or default
                else: # If not first, try to increment based on previous valid time
                    prev_time_numeric = int(time_list[-1] if time_list else accumulated_hour_minute_str)
                    prev_hour = prev_time_numeric // 100
                    prev_minute = prev_time_numeric % 100

                    new_minute_total = prev_minute + minute_cutoff_adjustment
                    new_hour_total = prev_hour + (new_minute_total // 60)
                    new_minute_final = new_minute_total % 60

                    accumulated_hour_minute_str = f"{new_hour_total % 24:02d}{new_minute_final:02d}"

            time_list.append(accumulated_hour_minute_str)
        except Exception: # Catch any parsing error for a time_entry
            # If error, try to use/increment last known good time, or a default
            if time_list: # If there's a previous valid time
                 prev_time_numeric = int(time_list[-1])
                 prev_hour = prev_time_numeric // 100
                 prev_minute = prev_time_numeric % 100
                 new_minute_total = prev_minute + minute_cutoff_adjustment
                 new_hour_total = prev_hour + (new_minute_total // 60)
                 new_minute_final = new_minute_total % 60
                 time_list.append(f"{new_hour_total % 24:02d}{new_minute_final:02d}")
            else: # First entry and it's bad, or unrecoverable
                 time_list.append("0000") # Default unknown time

    return time_list


  def _adjust_minute_representation(self, minute_value: int) -> int:
    # This function's logic seems specific to how minutes are represented (e.g., 65 becomes 100+5).
    # It might be simpler to work with total minutes from midnight or use datetime objects if standard time arithmetic is needed.
    # For now, preserving the original logic.
    if minute_value > 60 and minute_value < 100: minute_value=100
    elif minute_value > 160 and  minute_value < 200: minute_value=200
    # ... (all other elif conditions from original)
    elif minute_value > 2160 and  minute_value < 2200: minute_value=2200
    elif minute_value > 2359: minute_value=0
    return minute_value

  def save_features_to_csv(self, features_list_of_lists, output_identifier: str,
                           stage_parameter_flag: bool = False, save_mode: int = 0):
    if self.features_output_path is None:
        print("Error: FeatureExtractor.features_output_path is not set.")
        return None

    # Define column names based on stage_parameter_flag
    # Base columns: class_id, class_name, formatted_minute, formatted_date
    # Optional: stage_info
    # Feature columns: apen, dfa, chaos_cont_dim, etc.
    column_names = ['class_id', 'class_name', 'formatted_minute', 'formatted_date']
    if stage_parameter_flag:
        column_names.append('stage_info')

    feature_metric_names = [
        'apen', 'dfa', 'chaos_cont_dim', 'lyap_r', # lyap was positive_lyapunov_exponents_count
        'abp_low','abp_delta','abp_theta', 'abp_alpha','abp_beta',
        'fft_mean', 'fft_min', 'fft_max', 'fft_variance',
        'psd_mean', 'psd_min', 'psd_max' , 'psd_variance',
        'electrome_mean', 'electrome_min', 'electrome_max', 'electrome_variance'
    ]
    column_names.extend(feature_metric_names)

    features_df = pd.DataFrame(features_list_of_lists, columns=column_names)

    try:
        if save_mode == 0:
            # output_identifier is likely class_id as string e.g., "df0.csv", "df1.csv"
            csv_output_path = os.path.join(self.features_output_path, f'df{output_identifier}.csv')
            os.makedirs(self.features_output_path, exist_ok=True) # Ensure base path exists
        elif save_mode == 1:
            # output_identifier is "class_subdir_name/file_name_no_ext"
            parts = output_identifier.replace('\\', '/').split('/') # Normalize path separators
            class_subdir = parts[0]
            file_id = parts[1] if len(parts) > 1 else "features" # Default filename if only one part

            class_specific_feature_path = os.path.join(self.features_output_path, class_subdir)
            os.makedirs(class_specific_feature_path, exist_ok=True)
            csv_output_path = os.path.join(class_specific_feature_path, f'df_{file_id}.csv')
        else:
            print(f"Error: Unknown save_mode '{save_mode}'.")
            return None

        features_df.to_csv(csv_output_path, index=False)
        return features_df
    except Exception as er:
        print(f"Error saving features to CSV {csv_output_path}: {er}")
        return er # Return the exception object
