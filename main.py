"""
LacevApp - Application for processing and analysis of microvoltage data.

This module implements a graphical interface for processing and analysis of microvoltage data,
allowing visualization and analysis of different types of graphics.
"""

import warnings
import os
import glob
import pandas as pd
from typing import List, Dict, Any, Optional
from tkinter import OptionMenu, StringVar, ttk, filedialog, messagebox
import tkinter as tk
from PIL import Image, ImageTk
from ttkthemes import ThemedTk

from lacev_lib.app_processor import LacevAppProcessor
from lacev_lib.signal_filter import SignalFilter
from lacev_lib.feature_extractor import FeatureExtractor
from lacev_lib.signal_plotter import SignalPlotter

# Disable warnings for a better user experience
warnings.filterwarnings('ignore')

class LacevAppGUI:
    """
    Main class for the LacevApp GUI.
    Handles user interactions, orchestrates data processing and plotting
    by interacting with classes from the lacev_lib.
    """

    def __init__(self):
        """
        Initializes the main application window, variables, and widgets.
        Also initializes processor and utility class instance variables to None.
        """
        self.root = ThemedTk(theme='darkly')
        self.root.title('LacevApp v0.3 Refactored') # Updated title
        self.root.iconbitmap('lacevapp.ico') # Assuming lacevapp.ico is in the same directory or path

        # To store instances of helper classes, initialized when a directory is processed
        self.app_processor: Optional[LacevAppProcessor] = None
        self.signal_filter: Optional[SignalFilter] = None
        self.feature_extractor: Optional[FeatureExtractor] = None
        self.plot_handler: Optional[SignalPlotter] = None

        self.setup_variables()
        self.create_widgets()

    def setup_variables(self) -> None:
        """
        Configures Tkinter StringVars for dynamic UI text.
        """
        self.plot_type_var = StringVar(value='Select Plot Type') # Initial value changed
        self.instruction_text_var = tk.StringVar()
        self.instruction_text_var.set("LacevApp needs significant processing power. Close other programs for best performance.")
        self.browse_button_text_var = tk.StringVar(value='Open Data Folder')

    def create_widgets(self) -> None:
        """
        Creates and positions all static UI widgets in the main window.
        """
        # Dropdown menu for plot types
        plot_options = ["FFT-PSD", "TDAF", "TDAFLog"] # Default options
        self.plot_type_dropdown = OptionMenu(
            self.root,
            self.plot_type_var,
            plot_options[0], # Default selection
            *plot_options,
            command=self.handle_plot_selection
        )
        self.plot_type_dropdown.config(font="Raleway", bg='green', fg='white', height=2, width=15)
        self.plot_type_dropdown.grid(column=0, row=0, padx=10, pady=10)

        # Logo
        self.setup_logo()

        # Progress bar
        self.progress_bar = ttk.Progressbar(
            self.root,
            orient='horizontal',
            mode='determinate', # Can be set to 'indeterminate' for long, non-specific tasks
            length=500
            # maximum is 100 by default
        )
        self.progress_bar.grid(column=1, row=1, padx=10, pady=10)

        # Instruction label
        self.instruction_label = tk.Label(
            self.root,
            textvariable=self.instruction_text_var,
            font="Raleway",
            bg='red', # Consider a less alarming color or theme-based
            fg='white'
        )
        self.instruction_label.grid(columnspan=3, column=0, row=3, padx=10, pady=10, sticky="ew") # Spans more columns

        # Browse button
        self.browse_button = tk.Button(
            self.root,
            textvariable=self.browse_button_text_var,
            command=self.open_directory,
            font="Raleway",
            bg='#007bff', # A more modern blue
            fg='white',
            height=2,
            width=15,
            relief=tk.RAISED, # Added relief
            bd=2 # Added border
        )
        self.browse_button.grid(column=2, row=0, padx=10, pady=10) # Positioned next to dropdown

        # Log text area (created here, configured in setup_log_text_area)
        log_frame = ttk.Frame(self.root, width=400, height=150)
        log_frame.grid(column=0, row=2, columnspan=3, padx=10, pady=10, sticky="nsew")
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(0, weight=1)

        scrollbar_y = ttk.Scrollbar(log_frame, orient=tk.VERTICAL)
        self.log_text_area = tk.Text(log_frame, height=10, width=60, padx=5, pady=5,
                                     yscrollcommand=scrollbar_y.set, relief=tk.SOLID, bd=1)
        self.log_text_area.grid(row=0, column=0, sticky="nsew")
        scrollbar_y.config(command=self.log_text_area.yview)
        scrollbar_y.grid(row=0, column=1, sticky="ns")

        self.root.grid_columnconfigure(1, weight=1) # Allow middle column (logo, progress) to expand
        self.root.grid_rowconfigure(2, weight=1) # Allow log area row to expand

    def setup_logo(self) -> None:
        """Configures and displays the application logo, if found."""
        try:
            logo_image = Image.open('logo.png')
            # Consider resizing logo if it's too large
            # logo_image = logo_image.resize((width, height), Image.ANTIALIAS)
            self.logo_image_tk = ImageTk.PhotoImage(logo_image) # Store on self to prevent garbage collection
            logo_label = tk.Label(self.root, image=self.logo_image_tk)
            # logo_label.image = self.logo_image_tk # Not needed if self.logo_image_tk is used directly
            logo_label.grid(column=1, row=0, padx=10, pady=10)
        except FileNotFoundError:
            self.log_message("Logo file 'logo.png' not found.")
        except Exception as e:
            messagebox.showerror("Logo Error", f"Could not load logo: {str(e)}")

    def log_message(self, message: str, show_ messagebox_type: Optional[str] = None, title: str = "Info"):
        """Helper function to log messages to the text area and optionally show a messagebox."""
        if hasattr(self, 'log_text_area') and self.log_text_area:
            self.log_text_area.insert(tk.END, message + "\n")
            self.log_text_area.see(tk.END)
        else:
            print(message) # Fallback to console if text area not ready

        if show_messagebox_type:
            if show_messagebox_type == "error":
                messagebox.showerror(title, message)
            elif show_messagebox_type == "warning":
                messagebox.showwarning(title, message)
            elif show_messagebox_type == "info":
                messagebox.showinfo(title, message)
        self.root.update_idletasks()


    def open_directory(self) -> None:
        """
        Handles the 'Open Folder' button action.
        Prompts the user to select a directory and then initiates data processing.
        """
        self.instruction_text_var.set("Selecting directory...")

        selected_directory = filedialog.askdirectory()
        if not selected_directory:
            self.instruction_text_var.set("No directory selected. Please select a data folder.")
            return # User cancelled or selected no directory

        self.root.directory = selected_directory # Store for potential future use by plotting
        self.log_message(f"Directory selected: {self.root.directory}")
        self.instruction_text_var.set(f"Processing directory: {os.path.basename(self.root.directory)}...")
        if self.browse_button: # It might have been destroyed in a previous run
             self.browse_button.config(state=tk.DISABLED) # Disable during processing

        self.process_directory_data()

        if self.browse_button: # Re-enable if it wasn't destroyed
            self.browse_button.config(state=tk.NORMAL)
        self.instruction_text_var.set("Processing complete. Select a plot or open another folder.")


    def process_directory_data(self) -> None:
        """
        Orchestrates the data processing workflow:
        1. Initializes LacevAppProcessor.
        2. Creates output directories.
        3. Initializes SignalFilter and FeatureExtractor.
        4. Processes all class files found by app_processor.
        5. Opens the features directory on completion.
        """
        try:
            self.app_processor = LacevAppProcessor(self.root.directory)
            if self.app_processor.initialization_status['error']:
                self.log_message(f"Initialization Error: {self.app_processor.initialization_status['message']}", "error", "Initialization Failed")
                return

            self.log_message("LacevAppProcessor initialized.")
            dir_creation_status = self.app_processor.create_output_directories()
            for msg in dir_creation_status['messages']: self.log_message(msg)
            if dir_creation_status['error']:
                self.log_message("Directory creation encountered errors. Processing may fail.", "warning", "Directory Issues")
                # Optionally, ask user if they want to proceed or return

            # Initialize signal_filter and feature_extractor here
            self.signal_filter = SignalFilter(filtered_data_output_path=self.app_processor.get_filtered_data_path())
            self.feature_extractor = FeatureExtractor(features_output_path=self.app_processor.get_features_path())
            self.log_message("SignalFilter and FeatureExtractor initialized.")

            self.process_all_class_files() # Removed app_processor argument, uses self.app_processor

            self.log_message("All processing complete. Opening features directory...")
            explore_status = self.app_processor.open_path_in_explorer(self.app_processor.get_features_path())
            if explore_status['error']:
                self.log_message(f"Explorer Error: {explore_status['message']}", "warning", "File Explorer")

        except Exception as e:
            self.log_message(f"Critical error during directory processing: {str(e)}", "error", "Processing Error")

    def setup_log_text_area(self, text_messages: List[str]) -> None:
        """
        Populates the log text area with initial messages.
        This method is called early, but actual logging should use self.log_message.
        """
        if not hasattr(self, 'log_text_area'): # Ensure text area exists
            print("Log text area not ready for setup messages.")
            return

        for message in text_messages:
            self.log_message(message) # Use the helper
            # Progress bar updates here based on setup messages might not be very informative
            # self.progress_bar['value'] += (100 / len(text_messages)) if text_messages else 0
            # self.root.update_idletasks()

        # Reset progress bar after initial messages if it was used
        # self.progress_bar['value'] = 0
        # self.root.update_idletasks()

    def process_all_class_files(self) -> None: # Removed app_processor, uses self.app_processor
        """
        Iterates through class subdirectories and processes files within them.
        Uses self.app_processor, self.signal_filter, self.feature_extractor.
        """
        if not self.app_processor or not self.signal_filter or not self.feature_extractor:
            self.log_message("Error: Processors not initialized. Cannot process class files.", "error", "Processing Error")
            return

        class_id = 0
        minute_segment_length = 3 # This could be a configurable parameter

        class_subdirs = self.app_processor.get_class_subdir_names()
        if not class_subdirs:
            self.log_message("No class subdirectories found to process.", "warning", "Data Error")
            return

        total_classes = len(class_subdirs)
        self.progress_bar['maximum'] = total_classes # Progress per class
        self.progress_bar['value'] = 0

        for class_subdir_name in class_subdirs:
            self.log_message(f"Processing class: {class_subdir_name}...")
            self.process_files_in_class_subdir(class_subdir_name, class_id, minute_segment_length) # app_processor, filter, extractor removed
            class_id += 1
            self.progress_bar['value'] = class_id
            self.root.update_idletasks()

    def process_files_in_class_subdir(self, class_subdir_name: str, class_id: int, minute_segment_length: int) -> None:
        """
        Processes all valid data files within a specific class subdirectory.
        Uses self.app_processor, self.signal_filter, self.feature_extractor.
        """
        if not self.app_processor or not self.signal_filter or not self.feature_extractor: # Should not happen if called from process_all_class_files
            self.log_message("Error: Processors not initialized.", "error", "Internal Error")
            return

        class_subdir_full_path = os.path.join(self.app_processor.get_raw_data_path(), class_subdir_name)

        try:
            directory_listing = [f for f in os.listdir(class_subdir_full_path) if os.path.isfile(os.path.join(class_subdir_full_path, f))]
        except FileNotFoundError:
            self.log_message(f"Error: Raw data subdirectory not found: {class_subdir_full_path}", "error", "File Error")
            return
        except Exception as e:
            self.log_message(f"Error listing files in {class_subdir_full_path}: {e}", "error", "File Error")
            return

        if not directory_listing:
            self.log_message(f"No files found in {class_subdir_full_path}.", "warning")
            return

        # For overall progress within this class subdir (can be refined)
        # total_files_in_subdir = len(directory_listing)
        # current_file_num = 0

        # This results_list is for features of *all* files in this class subdir, to be aggregated
        class_features_results_list = []

        for file_item in directory_listing:
            file_name_no_ext, file_extension = os.path.splitext(str(file_item))

            if file_extension.lower() in ['.csv', '.txt']: # Make extension check case-insensitive
                self.log_message(f"Processing file: {file_item} in {class_subdir_name}...")
                # This list is for features of the *current* file, to be saved per file
                single_file_features_list = []
                self.process_single_data_file(
                    class_subdir_full_path, file_item, file_name_no_ext, class_subdir_name, class_id,
                    minute_segment_length, single_file_features_list # Pass the list for this single file
                )
                # After processing a single file, its features are in single_file_features_list.
                # These are saved per file by feature_extractor.save_features_to_csv(mode=1)
                # For aggregation, we'd need to append to class_features_results_list if that's the strategy.
                # The current aggregate_and_save_class_features reads from those saved files.
                # So, results_list passed to process_single_data_file is only for that file's segments.

                # current_file_num += 1
                # Update progress within the class if needed
                # self.progress_bar['value'] = (current_file_num / total_files_in_subdir) * 100 (if progress is per file)

        # Aggregation is handled after all files in the class subdir are processed
        self.aggregate_and_save_class_features(class_subdir_name, class_id)

    def process_single_data_file(self, class_subdir_full_path: str, file_item: str,
                          file_name_no_ext: str, class_subdir_name: str, class_id: int, minute_segment_length: int,
                          current_file_results_list: List) -> None: # Renamed results_list for clarity
        """
        Processes a single data file: reads, filters, extracts features, and saves per-file features.
        Uses self.signal_filter, self.feature_extractor.
        """
        try:
            full_file_path = os.path.join(class_subdir_full_path, file_item)
            self.log_message(f"Reading: {file_item}")
            raw_data_df = pd.read_csv(
                full_file_path, encoding='utf-8', delimiter=",", engine='c',
                low_memory=False, memory_map=True,
            )

            if raw_data_df.empty or len(raw_data_df.columns) == 0 or len(raw_data_df.iloc[:, 0]) == 0:
                self.log_message(f"Warning: File {file_name_no_ext} is empty or has no data/columns. Skipping.", "warning")
                return

            # Filter data
            self.log_message(f"Filtering: {file_name_no_ext}")
            processed_signal_data = self.signal_filter.apply_filters_and_save(
                raw_signal_data=raw_data_df,
                output_subdir_name=class_subdir_name,
                output_file_name_prefix=file_name_no_ext
            )
            if processed_signal_data is None or len(processed_signal_data) < 2:
                self.log_message(f"Error filtering {file_name_no_ext}. Skipping feature extraction.", "error")
                return
            filtered_microvolts = processed_signal_data[1]

            # Process time and date from raw_data_df as they are metadata related to the raw signal
            time_data = self.process_time(raw_data_df, minute_segment_length)
            date_data = self.process_date(raw_data_df)

            # Extract features
            self.log_message(f"Extracting features for: {file_name_no_ext}")
            # current_file_results_list is modified in place by extract_features
            self.feature_extractor.extract_features(
                data_array=filtered_microvolts,
                segment_duration_minutes=minute_segment_length,
                start_time_info=time_data, start_date_info=date_data,
                class_id=class_id, class_name=class_subdir_name,
                sample_rate=60, # TODO: Make sample_rate dynamic or configurable
                feature_results_list=current_file_results_list, # This list will hold features for segments of this file
                file_name_prefix=file_name_no_ext
            )
            self.log_message(f"Features extracted for {file_name_no_ext}.")

            # Save features for this single file (mode=1 behavior)
            # The output_identifier for mode=1 is "class_subdir_name/file_name_no_ext"
            output_identifier_mode1 = os.path.join(class_subdir_name, file_name_no_ext)
            save_status = self.feature_extractor.save_features_to_csv(
                current_file_results_list,
                output_identifier_mode1,
                save_mode=1
            )
            if isinstance(save_status, Exception) or save_status is None : # save_features_to_csv returns DataFrame on success, Exception on error
                self.log_message(f"Error saving per-file features for {file_name_no_ext}: {save_status}", "error")
            else:
                self.log_message(f"Per-file features saved for {file_name_no_ext}.")

        except pd.errors.EmptyDataError:
            self.log_message(f"Warning: File {full_file_path} is empty or unreadable. Skipping.", "warning")
        except FileNotFoundError:
            self.log_message(f"Error: File not found {full_file_path}. Skipping.", "error")
        except Exception as e:
            self.log_message(f"Critical error processing file {file_name_no_ext}: {str(e)}", "error", "File Processing Error")

    def process_time(self, raw_data_df: pd.DataFrame, minute_segment_length: int) -> List[str]: # Return type changed
        """Processes the time data from 'HORA' column."""
        if 'HORA' in raw_data_df.columns:
            try:
                # Ensure feature_extractor is initialized
                if self.feature_extractor:
                     return self.feature_extractor.format_time_from_hora_column(raw_data_df, minute_segment_length)
                else: # Should not happen if process_directory_data initializes it
                    self.log_message("Feature extractor not initialized for time processing.", "error")
                    return ["0000"] * len(raw_data_df) # Return default list of appropriate length
            except Exception as e:
                self.log_message(f"Error formatting time: {e}", "warning")
                return ["0000"] * len(raw_data_df) # Fallback
        return ["0000"] * len(raw_data_df) # Default if 'HORA' not present

    def process_date(self, raw_data_df: pd.DataFrame) -> List[str]:
        """Processes the date data from 'DATA' or 'DATA ' column."""
        date_column_name = 'DATA' if 'DATA' in raw_data_df.columns else 'DATA '
        if date_column_name in raw_data_df.columns:
            try:
                return [
                    '/'.join(str(part).replace(' ', '').replace('\\', '').replace('n', '').strip() for part in str(date_val).split('-'))
                    for date_val in raw_data_df[date_column_name]
                ]
            except Exception as e:
                self.log_message(f"Error formatting date: {e}", "warning")
                return ["00/00/0000"] * len(raw_data_df) # Fallback
        return ["00/00/0000"] * len(raw_data_df) # Default if no date column

    def aggregate_and_save_class_features(self, class_subdir_name: str, class_id: int) -> None:
        """
        Aggregates individual feature CSVs (saved per file) for a given class
        into a single CSV file for that class. Uses self.app_processor.
        """
        if not self.app_processor or not self.feature_extractor : # Ensure processor and extractor are available
            self.log_message("Error: AppProcessor or FeatureExtractor not initialized for aggregation.", "error")
            return

        self.log_message(f"Aggregating features for class: {class_subdir_name} (ID: {class_id})...")
        class_specific_features_dir = os.path.join(self.app_processor.get_features_path(), class_subdir_name)

        try:
            all_feature_files_in_subdir = glob.glob(os.path.join(class_specific_features_dir, "*.csv"))
            if not all_feature_files_in_subdir:
                self.log_message(f"No individual feature CSVs found in {class_specific_features_dir} to aggregate for class {class_id}.", "warning")
                return

            list_of_dfs_for_class = []
            for feature_file_path in all_feature_files_in_subdir:
                try:
                    df_temp = pd.read_csv(feature_file_path)
                    list_of_dfs_for_class.append(df_temp)
                except pd.errors.EmptyDataError:
                    self.log_message(f"Warning: Feature file {feature_file_path} is empty. Skipping for aggregation.", "warning")
                except Exception as e:
                    self.log_message(f"Error reading feature file {feature_file_path} for aggregation: {e}", "error")

            if not list_of_dfs_for_class:
                self.log_message(f"No valid feature dataframes to aggregate for class {class_id}.", "warning")
                return

            aggregated_class_df = pd.concat(list_of_dfs_for_class, axis=0, ignore_index=True)

            # Save the final aggregated CSV for the class in the main features directory
            # This uses the save_features_to_csv method with mode=0 logic (identifier is class_id)
            save_status = self.feature_extractor.save_features_to_csv(
                features_list_of_lists=aggregated_class_df, # Pass DataFrame to be saved
                output_identifier=str(class_id),
                save_mode=0 # mode=0 saves as df<class_id>.csv in base features_output_path
            )
            if isinstance(save_status, Exception) or save_status is None:
                 self.log_message(f"Error saving aggregated features for class {class_id}: {save_status}", "error")
            else:
                self.log_message(f"Aggregated features for class {class_id} saved successfully.", "info")

        except Exception as e:
            self.log_message(f"Critical error during feature aggregation for class {class_id}: {str(e)}", "error", "Aggregation Error")

    def handle_plot_selection(self, selected_plot_type: str) -> None:
        """
        Handles the plot type selection from the dropdown menu.
        Re-initializes app_processor if a new directory might be at play or uses existing one.
        """
        self.instruction_text_var.set(f"Preparing to plot {selected_plot_type}...")
        # If directory hasn't been processed, or to allow re-selection for plotting
        if not self.root.directory: # Or if you want to always ask for plotting directory
            self.log_message("Please open a data directory first before plotting.", "info", "Directory Not Set")
            current_dir = filedialog.askdirectory()
            if not current_dir:
                self.instruction_text_var.set("Plotting cancelled: No directory selected.")
                return
            self.root.directory = current_dir
            # If a new directory is selected for plotting, we might need to re-initialize app_processor
            # For simplicity now, assume process_directory_data was run for the self.root.directory
            # or that plotting only uses existing processed data.
            # If app_processor is specific to a processing run, it might need to be re-instantiated.
            try:
                self.app_processor = LacevAppProcessor(self.root.directory)
                if self.app_processor.initialization_status['error']:
                    self.log_message(self.app_processor.initialization_status['message'], "error", "Initialization Failed")
                    return
            except Exception as e:
                 self.log_message(f"Error re-initializing for plotting: {e}", "error")
                 return

        if not self.app_processor: # Should be set if directory was processed
            self.log_message("Data directory not processed yet. Please process data first.", "error", "Processing Required")
            return

        # Initialize graph directories (this also checks for other output dirs)
        graph_dir_status = self.app_processor.initialize_graph_directories()
        for msg in graph_dir_status['messages']: self.log_message(msg)
        if graph_dir_status['error']:
             # Allow plotting to proceed but warn user.
            self.log_message("Graph directory status check reported issues. Plotting may fail.", "warning")

        self.execute_plot_action(selected_plot_type) # Removed app_processor, uses self.app_processor

    def execute_plot_action(self, selected_plot_type: str) -> None: # Removed app_processor
        """
        Executes the selected plotting action using SignalPlotter.
        Uses self.app_processor.
        """
        if not self.app_processor:
            self.log_message("Cannot execute plot: AppProcessor not available.", "error", "Plotting Error")
            return

        self.instruction_text_var.set(f"Loading plot: {selected_plot_type}...")
        if self.browse_button and self.browse_button.winfo_exists():
            self.browse_button.config(state=tk.DISABLED) # Disable browse during plotting

        # Instantiate SignalPlotter with necessary paths from self.app_processor
        try:
            self.plot_handler = SignalPlotter(
                graphics_output_path=self.app_processor.get_graphics_path(),
                features_input_path=self.app_processor.get_features_path(),
                raw_data_input_path=self.app_processor.get_raw_data_path(),
                filtered_data_input_path=self.app_processor.get_filtered_data_path()
            )
            self.log_message("SignalPlotter initialized for plotting.")
        except Exception as e:
            self.log_message(f"Error initializing SignalPlotter: {e}", "error", "Plotting Setup Error")
            if self.browse_button and self.browse_button.winfo_exists():
                self.browse_button.config(state=tk.NORMAL)
            return

        class_subdir_names = self.app_processor.get_class_subdir_names()

        plot_method = None
        if selected_plot_type == 'FFT-PSD':
            plot_method = self.plot_handler.plot_fft_psd_for_file
        elif selected_plot_type == 'TDAF':
            plot_method = self.plot_handler.plot_tdaf_analysis
        elif selected_plot_type == 'TDAFLog':
            plot_method = self.plot_handler.plot_tdaf_analysis_log_scale

        plot_result = None # To store dict from plotting method
        if plot_method:
            try:
                self.log_message(f"Generating {selected_plot_type} plot(s)...")
                # sample_rate is hardcoded for now, consider making it configurable
                plot_result = plot_method(class_subdir_names_list=class_subdir_names, sample_rate=50)

                if plot_result and plot_result.get('error'):
                    self.log_message(f"Error generating {selected_plot_type} plot: {plot_result.get('message', 'Unknown plotting error')}", "error", "Plotting Error")
                else:
                    self.log_message(f"{selected_plot_type} plot(s) generated successfully. Path: {plot_result.get('path', 'N/A') if plot_result else 'N/A'}", "info")
                    explore_status = self.app_processor.open_path_in_explorer(self.app_processor.get_graphics_path())
                    if explore_status['error']:
                        self.log_message(explore_status['message'], "warning", "File Explorer")
            except Exception as e:
                self.log_message(f"Critical error during {selected_plot_type} plotting: {e}", "error", "Plotting Error")
        else:
            self.log_message(f"Selected plot type '{selected_plot_type}' is not recognized.", "warning", "Plotting Error")

        if self.browse_button and self.browse_button.winfo_exists():
            self.browse_button.config(state=tk.NORMAL) # Re-enable browse button
        self.instruction_text_var.set(f"Plotting for {selected_plot_type} finished. Select another action.")

    def run(self) -> None:
        """Starts the Tkinter main event loop."""
        self.root.mainloop()

if __name__ == "__main__":
    app_gui = LacevAppGUI()
    app_gui.run()

[end of main.py]
