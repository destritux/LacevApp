import os
import subprocess

class LacevAppProcessor:
    def __init__(self, root_directory: str):
        self.root_directory = root_directory
        self.raw_data_path = None
        self.filtered_data_path = None
        self.ml_results_path = None
        self.graphics_path = None
        self.features_path = None
        self.class_subdir_names = []

        # Call initialize_paths to set up all path variables and class_subdir_names
        self.initialization_status = self.initialize_paths()

    def initialize_paths(self) -> dict:
        """
        Initializes all necessary directory paths based on the root_directory.
        Identifies class subdirectories within the raw data path.
        """
        if not self.root_directory or not os.path.isdir(self.root_directory):
            return {'error': True, 'message': f"Root directory '{self.root_directory}' not found or is not a directory."}

        self.raw_data_path = os.path.join(self.root_directory, 'raw')
        self.filtered_data_path = os.path.join(self.root_directory, 'filtered')
        self.ml_results_path = os.path.join(self.root_directory, 'mlResults')
        self.graphics_path = os.path.join(self.root_directory, 'graphics')
        self.features_path = os.path.join(self.root_directory, 'features')

        if not os.path.isdir(self.raw_data_path):
            # Try to create raw_data_path if it doesn't exist, as it's fundamental
            try:
                os.mkdir(self.raw_data_path)
            except OSError as e:
                 return {'error': True, 'message': f"Raw data directory '{self.raw_data_path}' not found and could not be created: {e}"}

        try:
            self.class_subdir_names = [
                f for f in os.listdir(self.raw_data_path)
                if os.path.isdir(os.path.join(self.raw_data_path, f))
            ]
            return {'error': False, 'message': 'Paths initialized successfully.'}
        except FileNotFoundError:
            # This case might be redundant if the raw_data_path check above is solid
            return {'error': True, 'message': f"Raw data directory '{self.raw_data_path}' not found when trying to list subdirectories."}
        except Exception as e:
            return {'error': True, 'message': f"An unexpected error occurred while listing subdirectories: {e}"}


    def create_output_directories(self) -> dict:
        """
        Creates the main output directories and class-specific subdirectories
        in 'filtered' and 'features' paths.
        Returns a dictionary with 'error' status and a list of messages.
        """
        messages = []
        has_errors = False

        if not self.class_subdir_names and os.path.isdir(self.raw_data_path):
            # Attempt to populate class_subdir_names if empty but raw_data_path exists
            try:
                self.class_subdir_names = [
                    f for f in os.listdir(self.raw_data_path)
                    if os.path.isdir(os.path.join(self.raw_data_path, f))
                ]
            except Exception as e:
                messages.append(f"Error re-listing raw subdirectories: {e}")
                # Proceeding without class_subdir_names if listing fails again

        if not self.class_subdir_names: # Check again after potential re-population
             messages.append('Warning: No class subdirectories found or specified in raw data path. Only main output directories will be created.')
             # Depending on requirements, this could be an error. For now, proceeding with main dirs.

        main_output_dirs = {
            "filtered": self.filtered_data_path,
            "mlResults": self.ml_results_path,
            "graphics": self.graphics_path,
            "features": self.features_path
        }

        for dir_key, path_val in main_output_dirs.items():
            if path_val is None: # Should not happen if __init__ and initialize_paths worked
                messages.append(f"Error: Path for '{dir_key}' is not initialized.")
                has_errors = True
                continue
            try:
                os.makedirs(path_val, exist_ok=True) # Use makedirs with exist_ok=True
                messages.append(f"Directory '{path_val}' ensured to exist.")
            except OSError as e:
                messages.append(f"Error creating directory '{path_val}': {e}")
                has_errors = True

        # Create class-specific subdirectories
        for subdir_name in self.class_subdir_names:
            paths_to_create_for_subdir = {
                "filtered_class_subdir": os.path.join(self.filtered_data_path, subdir_name),
                "features_class_subdir": os.path.join(self.features_path, subdir_name)
            }
            for key, class_path_val in paths_to_create_for_subdir.items():
                try:
                    os.makedirs(class_path_val, exist_ok=True)
                    messages.append(f"Directory '{class_path_val}' ensured to exist.")
                except OSError as e:
                    messages.append(f"Error creating class subdirectory '{class_path_val}': {e}")
                    has_errors = True

        return {'error': has_errors, 'messages': messages}

    def initialize_graph_directories(self) -> dict:
        """
        Ensures that directories required for saving graphs exist.
        This method is similar to create_output_directories but focuses on graphics and related paths.
        Considered merging, but keeping separate if distinct logic or checks are needed for graphics.
        """
        # Primarily, graphics_path needs to exist. Other paths are inputs.
        messages = []
        has_errors = False

        if not self.graphics_path:
            messages.append("Error: Graphics path is not initialized.")
            return {'error': True, 'messages': messages}

        try:
            os.makedirs(self.graphics_path, exist_ok=True)
            messages.append(f"Graphics directory '{self.graphics_path}' ensured to exist.")
        except OSError as e:
            messages.append(f"Error creating graphics directory '{self.graphics_path}': {e}")
            has_errors = True

        # Check if dependent input directories (raw, filtered, features) exist, as plots might need them.
        # These are not created by this method but checked for existence.
        input_paths_to_check = {
            "raw data": self.raw_data_path,
            "filtered data": self.filtered_data_path,
            "features data": self.features_path
        }
        for path_key, path_val in input_paths_to_check.items():
            if not path_val or not os.path.isdir(path_val):
                messages.append(f"Warning: Input directory for {path_key} ('{path_val}') not found. Plotting may fail.")
                # This is a warning, not necessarily an error for this method's primary purpose.

        return {'error': has_errors, 'messages': messages}


    def open_path_in_explorer(self, path_to_open: str):
        """Opens the given path in the system's file explorer (Windows specific)."""
        # This remains OS-specific. Consider cross-platform alternatives if needed.
        try:
            # Attempt to use os.startfile for a more direct approach on Windows
            if os.name == 'nt':
                os.startfile(path_to_open)
                return {'error': False, 'message': f"Attempted to open '{path_to_open}'."}
            else:
                # Fallback for non-Windows or if os.startfile isn't suitable
                filebrowser_path = os.path.join(os.getenv('WINDIR', ''), 'explorer.exe') # Default WINDIR if not set
                if not os.path.exists(filebrowser_path) and os.name == 'posix': # Basic Linux/macOS check
                    filebrowser_path = 'xdg-open' # Common on Linux
                    # For macOS, 'open' could also be used.

                normalized_path = os.path.normpath(path_to_open)
                if os.path.isdir(normalized_path):
                    subprocess.run([filebrowser_path, normalized_path], check=False) # check=False to not raise error on fail
                    return {'error': False, 'message': f"Attempted to open '{path_to_open}' with '{filebrowser_path}'."}
                else:
                    return {'error': True, 'message': f"Path '{path_to_open}' is not a valid directory."}
        except Exception as e:
            return {'error': True, 'message': f"Failed to open path '{path_to_open}': {e}"}

    # Getter methods
    def get_class_subdir_names(self) -> list:
        return self.class_subdir_names

    def get_raw_data_path(self) -> str:
        return self.raw_data_path

    def get_filtered_data_path(self) -> str:
        return self.filtered_data_path

    def get_ml_results_path(self) -> str:
        return self.ml_results_path

    def get_graphics_path(self) -> str:
        return self.graphics_path

    def get_features_path(self) -> str:
        return self.features_path

    # Convenience method to get all paths, useful for other classes
    def get_all_paths(self) -> dict:
        return {
            "root": self.root_directory,
            "raw": self.raw_data_path,
            "filtered": self.filtered_data_path,
            "ml_results": self.ml_results_path,
            "graphics": self.graphics_path,
            "features": self.features_path
        }
