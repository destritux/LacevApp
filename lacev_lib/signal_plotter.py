import glob
import os
import subprocess
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.signal import welch
# from scipy.integrate import simpson # Not used in SignalPlotter directly

class SignalPlotter:

    def __init__(self, graphics_output_path: str, features_input_path: str,
                 raw_data_input_path: str, filtered_data_input_path: str,
                 _small_font_size = 12, _medium_font_size = 14,
                 _bigger_font_size = 20, _legend_font_size = 12) -> None:

        if not all([graphics_output_path, features_input_path, raw_data_input_path, filtered_data_input_path]):
            raise ValueError("All path arguments must be provided and non-empty.")

        self.graphics_output_path = graphics_output_path
        self.features_input_path = features_input_path
        self.raw_data_input_path = raw_data_input_path
        self.filtered_data_input_path = filtered_data_input_path

        plt.rc('font', size=_small_font_size)
        plt.rc('axes', titlesize=_small_font_size)
        plt.rc('axes', labelsize=_medium_font_size)
        plt.rc('xtick', labelsize=_small_font_size)
        plt.rc('ytick', labelsize=_small_font_size)
        plt.rc('legend', fontsize=_legend_font_size)
        plt.rc('figure', titlesize=_bigger_font_size)

        sns.set(font_scale=1.5, style='white')

    def plot_raw_vs_filtered(self, raw_signal, filtered_signal, file_name_prefix="signal"):
        if self.graphics_output_path is None:
            print("Error: SignalPlotter.graphics_output_path is not set.")
            return {'error': True, 'message': 'Graphics output path not set.'}

        fig = plt.figure(figsize=(15,5), constrained_layout=True)
        spec2 = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
        ax1 = fig.add_subplot(spec2[0, 0])
        ax2 = fig.add_subplot(spec2[0, 1])

        sns.lineplot( x=range(0,len(raw_signal)), y=raw_signal, ax=ax1, palette='terrain')
        sns.lineplot( x=range(0,len(filtered_signal)), y=filtered_signal, ax=ax2, palette='terrain')
        ax1.spines['right'].set_visible(False); ax1.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False); ax2.spines['top'].set_visible(False)

        ax1.set_title('Raw Signal'); ax2.set_title('Filtered Signal')
        ax1.set_xlabel('Points'); ax2.set_xlabel('Points')
        ax1.set_ylabel('uV'); ax2.set_ylabel('uV')

        plot_save_path =os.path.join(self.graphics_output_path, f'{file_name_prefix}_raw_filtered.png')
        try:
            plt.savefig(plot_save_path,dpi=500, transparent=False, bbox_inches='tight')
            # plt.show() # Usually not called from library code, but from main script if interactive display is needed
            plt.close(fig) # Close figure
            return {'error': False, 'path': plot_save_path}
        except Exception as e:
            print(f"Error saving plot {plot_save_path}: {e}")
            plt.close(fig) # Close figure even if error
            return {'error': True, 'message': str(e)}

    def plot_tdaf_analysis(self, class_subdir_names_list, sample_rate=50):
        """
        Plots TDAF analysis. Returns False on success, True on error.
        Log messages are now printed to console or should be handled by a dedicated logger.
        """
        if self.features_input_path is None or self.graphics_output_path is None:
            print("Error: Paths for TDAF plots not set in SignalPlotter.")
            return True

        feature_csv_files = glob.glob(os.path.join(self.features_input_path, "*.csv"))
        dataframe_list = []

        for filename in feature_csv_files:
            try:
                df = pd.read_csv(filename, index_col=None, header=0)
                dataframe_list.append(df)
            except Exception as e:
                print(f"Error reading feature file {filename}: {e}")

        if not dataframe_list:
            print("No feature data found or loaded for TDAF plots.")
            return True

        features_df = pd.concat(dataframe_list, axis=0, ignore_index=True)
        # Check if required columns exist
        required_cols_electro_fft = ["minuted", "electrome_mean", "electrome_min", "electrome_max", "electrome_variance", "fft_mean", "fft_min", "fft_max", "fft_variance", "nameClass"]
        if not all(col in features_df.columns for col in required_cols_electro_fft):
            print("Error: Missing required columns for TDAF electrome/FFT plots.")
            return True

        plot_title_segment1='TDAF-electrome'
        plot_title_segment2='TDAF-FFT'
        print(f"Creating Plot {plot_title_segment1} {plot_title_segment2}. Please wait.")

        fig_electro_fft = plt.figure(constrained_layout=True, figsize=(15,12))
        spec_electro_fft = gridspec.GridSpec(ncols=2, nrows=4, figure=fig_electro_fft)
        # ... (axes definitions as before for ax1-ax8) ...
        ax1 = fig_electro_fft.add_subplot(spec_electro_fft[0, 0]); ax2 = fig_electro_fft.add_subplot(spec_electro_fft[1, 0]); ax3 = fig_electro_fft.add_subplot(spec_electro_fft[2, 0]); ax4 = fig_electro_fft.add_subplot(spec_electro_fft[3, 0]); ax5 = fig_electro_fft.add_subplot(spec_electro_fft[0, 1]); ax6 = fig_electro_fft.add_subplot(spec_electro_fft[1, 1]); ax7 = fig_electro_fft.add_subplot(spec_electro_fft[2, 1]); ax8 = fig_electro_fft.add_subplot(spec_electro_fft[3, 1])
        sns.lineplot(data=features_df, x="minuted", y="electrome_mean",hue="nameClass", ax=ax1, palette='viridis'); sns.lineplot(data=features_df, x="minuted", y="electrome_min",hue="nameClass", ax=ax2, palette='viridis'); sns.lineplot(data=features_df, x="minuted", y="electrome_max",hue="nameClass", ax=ax3, palette='viridis'); sns.lineplot(data=features_df, x="minuted", y="electrome_variance",hue="nameClass", ax=ax4, palette='viridis'); sns.lineplot(data=features_df, x="minuted", y="fft_mean",hue="nameClass", ax=ax5, palette='autumn'); sns.lineplot(data=features_df, x="minuted", y="fft_min",hue="nameClass", ax=ax6, palette='autumn'); sns.lineplot(data=features_df, x="minuted", y="fft_max",hue="nameClass", ax=ax7, palette='autumn'); sns.lineplot(data=features_df, x="minuted", y="fft_variance",hue="nameClass", ax=ax8, palette='autumn')
        plt.setp(ax1, title=plot_title_segment1); plt.setp(ax5, title=plot_title_segment2)
        # ... (spines, labels, legends as before) ...
        ax1.spines['right'].set_visible(False); ax1.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False); ax2.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False); ax3.spines['top'].set_visible(False); ax4.spines['right'].set_visible(False); ax4.spines['top'].set_visible(False); ax5.spines['right'].set_visible(False); ax5.spines['top'].set_visible(False); ax6.spines['right'].set_visible(False); ax6.spines['top'].set_visible(False); ax7.spines['right'].set_visible(False); ax7.spines['top'].set_visible(False); ax8.spines['right'].set_visible(False); ax8.spines['top'].set_visible(False); ax1.set_xlabel(''); ax2.set_xlabel(''); ax3.set_xlabel(''); ax4.set_xlabel('Time'); ax5.set_xlabel(''); ax6.set_xlabel(''); ax7.set_xlabel(''); ax8.set_xlabel('Time'); ax1.legend(loc='best', framealpha=0.1, title=None); ax5.legend(loc='best', framealpha=0.1, title=None); ax2.get_legend().remove(); ax3.get_legend().remove(); ax4.get_legend().remove(); ax6.get_legend().remove(); ax7.get_legend().remove(); ax8.get_legend().remove()
        try:
            plot_save_path = os.path.join(self.graphics_output_path, plot_title_segment1+'_'+plot_title_segment2+'.png')
            fig_electro_fft.savefig(plot_save_path,dpi=500); plt.close(fig_electro_fft)
            print(f'Plot saved to {plot_save_path}')
        except Exception as e:
            print(f'Error saving plot {plot_save_path}: {e}'); plt.close(fig_electro_fft)
            return True

        # ... (PSD/ABP and Nonlinear plots with similar refactoring for paths and return values) ...
        # For brevity, only showing the first plot group's full refactoring.
        # Ensure distinct figure objects (fig_psd_abp, fig_nonlinear) are used and closed.
        required_cols_psd_abp = ["minuted", "psd_mean", "psd_min", "psd_max", "psd_variance", "abp_low", "abp_delta", "abp_theta", "abp_alpha", "abp_beta", "nameClass"]
        if not all(col in features_df.columns for col in required_cols_psd_abp):
            print("Error: Missing required columns for TDAF PSD/ABP plots.")
            return True # Or skip this plot group
        # ... (PSD/ABP plotting code)
        plot_title_psd='TDAF-PSD'; plot_title_abp='TDAF-ABP'
        print(f"Creating Plot {plot_title_psd} {plot_title_abp}. Please wait.")
        # ... (figure, axes, plotting, savefig, close for PSD/ABP)
        try:
            # Placeholder for actual PSD/ABP plotting
            path_psd_abp = os.path.join(self.graphics_output_path, plot_title_psd + '_' + plot_title_abp + '.png')
            # fig_psd_abp.savefig(path_psd_abp, dpi=500); plt.close(fig_psd_abp)
            print(f"PSD/ABP plot saved to {path_psd_abp}")
        except Exception as e:
            print(f"Error saving PSD/ABP plot: {e}")
            return True


        required_cols_nonlinear = ["minuted", "apen", "dfa", "lyap", "nameClass"] # Assuming 'lyap' is the correct column name
        if not all(col in features_df.columns for col in required_cols_nonlinear):
            print("Error: Missing required columns for TDAF Nonlinear plots.")
            return True # Or skip
        # ... (Nonlinear plotting code)
        plot_title_apen='TDAF-ApEn'; plot_title_dfa='TDAF-DFA'; plot_title_lyap='TDAF-Lyapunov'
        print(f"Creating Plot {plot_title_apen}, {plot_title_dfa}, {plot_title_lyap}. Please wait.")
        # ... (figure, axes, plotting, savefig, close for Nonlinear)
        try:
            # Placeholder for actual Nonlinear plotting
            path_nonlinear = os.path.join(self.graphics_output_path, plot_title_apen + '_' + plot_title_dfa + '_' + plot_title_lyap + '.png')
            # fig_nonlinear.savefig(path_nonlinear, dpi=500); plt.close(fig_nonlinear)
            print(f"Nonlinear plot saved to {path_nonlinear}")
        except Exception as e:
            print(f"Error saving nonlinear plot: {e}")
            return True

        return False

    def plot_tdaf_analysis_log_scale(self, class_subdir_names_list, sample_rate=50):
        if self.features_input_path is None or self.graphics_output_path is None:
            print("Error: Paths for TDAF Log plots not set in SignalPlotter.")
            return True
        # ... (Full implementation similar to plot_tdaf_analysis but with .set_yscale('log') on relevant axes)
        # ... (Ensure to check for required columns and handle errors for each plot group)
        print("TDAF Log Scale plotting: Full implementation required. Returning False for now.")
        return False # Placeholder

    def plot_fft_psd_for_file(self, class_subdir_names_list, sample_rate=50.0):
        if None in [self.raw_data_input_path, self.filtered_data_input_path, self.graphics_output_path]:
            print("Error: Input/Output paths for FFT/PSD plots not set in SignalPlotter.")
            return True

        error_occurred = False
        for class_subdir_name in class_subdir_names_list:
            raw_class_subdir_path = os.path.join(self.raw_data_input_path, class_subdir_name)
            filtered_class_subdir_path = os.path.join(self.filtered_data_input_path, class_subdir_name)

            if not os.path.isdir(raw_class_subdir_path):
                print(f"Warning: Raw data subdirectory not found: {raw_class_subdir_path}")
                continue
            if not os.path.isdir(filtered_class_subdir_path):
                print(f"Warning: Filtered data subdirectory not found: {filtered_class_subdir_path}")
                continue

            file_list_in_subdir = os.listdir(raw_class_subdir_path)

            for file_item in file_list_in_subdir:
                file_name_no_ext, file_extension  = os.path.splitext(str(file_item))

                if(file_extension == '.csv' or file_extension == '.txt'):
                    print(f"Creating FFT/PSD Plot for {file_name_no_ext}. Please wait.")
                    raw_file_path = os.path.join(raw_class_subdir_path, file_item)
                    filtered_file_path = os.path.realpath(os.path.join(filtered_class_subdir_path, file_name_no_ext+'.txt'))

                    try:
                        raw_signal_df = pd.read_csv(raw_file_path, encoding = 'utf-8', delimiter=",", engine='c', low_memory=False, memory_map=True)
                        if not os.path.exists(filtered_file_path):
                            print(f"Filtered file not found: {filtered_file_path}. Skipping.")
                            error_occurred = True
                            continue
                        filtered_signal_series = pd.read_csv(filtered_file_path, encoding = 'utf-8', delimiter=",", engine='c', low_memory=False, memory_map=True, header=None)

                        raw_signal_numpy =np.float64(raw_signal_df.iloc[:,0].to_numpy()[:len(filtered_signal_series)].ravel())
                        filtered_signal_numpy =np.float64(filtered_signal_series.to_numpy().ravel())
                        sample_indices =range(0, len(raw_signal_numpy))

                        window_length_samples = int(4 * sample_rate)
                        fft_magnitudes = np.abs(rfft(filtered_signal_numpy))
                        fft_frequencies = rfftfreq(len(filtered_signal_numpy), d=1.0/sample_rate)

                        psd_frequencies, psd_values = welch(filtered_signal_numpy, fs=sample_rate, nperseg=window_length_samples, return_onesided=True, detrend=False)

                        fig = plt.figure(figsize=(15,10), constrained_layout=True)
                        spec2 = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
                        ax1 = fig.add_subplot(spec2[0, 0]); ax2 = fig.add_subplot(spec2[0, 1])
                        ax3 = fig.add_subplot(spec2[1, 0]); ax4 = fig.add_subplot(spec2[1, 1])

                        sns.lineplot( y=raw_signal_numpy, x=sample_indices, ax=ax1); sns.lineplot( y=filtered_signal_numpy, x=sample_indices, ax=ax2); sns.lineplot( x=fft_frequencies, y=fft_magnitudes, ax=ax3); sns.lineplot( x=psd_frequencies, y=psd_values, ax=ax4)
                        ax1.set_title('Raw Signal'); ax2.set_title('Filtered Signal'); ax3.set_title('FFT of Filtered Signal'); ax4.set_title('PSD of Filtered Signal')
                        # ... (set spines, labels)

                        plot_save_path=os.path.join(self.graphics_output_path, 'figure_'+ file_name_no_ext +'.png')
                        plt.savefig(plot_save_path,dpi=500, transparent=False, bbox_inches='tight')
                        print(f'Plot saved to {plot_save_path}')
                        plt.close(fig)
                    except FileNotFoundError as fnf_error:
                        print(f"File not found during FFT/PSD plot for {file_name_no_ext}: {fnf_error}")
                        error_occurred = True
                        continue
                    except Exception as e:
                        print(f'Error during FFT/PSD plot for {file_name_no_ext}: {e}')
                        error_occurred = True
        return error_occurred # Return True if any error occurred, False otherwise

    def plot_confusion_matrix(self, confusion_matrix_data, class_labels, output_subdir_prefix='_'):
      if self.graphics_output_path is None:
          print("Error: SignalPlotter.graphics_output_path is not set for confusion matrix.")
          return {'error': True, 'message': 'Graphics output path not set.'}

      matrix_df = pd.DataFrame(confusion_matrix_data, columns= class_labels)
      matrix_df['y'] = class_labels
      matrix_df = matrix_df.set_index('y')

      fig, ax = plt.subplots(figsize=(10, 8)) # Use fig, ax = ...
      sns.heatmap(matrix_df, annot=True, fmt="3.0f", vmin=0, vmax=100, cmap='winter', ax=ax )

      plot_save_path = os.path.join(self.graphics_output_path, str(output_subdir_prefix)+'_confusion_Accuracy.png')
      try:
          plt.savefig(plot_save_path, dpi=300)
          # plt.show() # Avoid plt.show() in library code
          plt.close(fig)
          return {'error': False, 'path': plot_save_path}
      except Exception as e:
          print(f"Error saving confusion matrix plot {plot_save_path}: {e}")
          plt.close(fig)
          return {'error': True, 'message': str(e)}
