# LacevApp

A desktop application for processing and analyzing microvoltage data. It allows users to load data, apply filters, extract features, and visualize the results through various plots.

## How to Use

The main UI elements and their functions are:

*   **"Open Folder" button:** To select the directory containing the raw data files.
*   **Dropdown menu (Ploats, FFT-PSD, TDAF, TDAFLog):** To select different types of plots for visualization after data processing.
*   **Progress bar:** Shows the progress of data processing.
*   **Text area:** Displays logs and messages during processing.

## Project Structure

*   `main.py`: Contains the main application logic and GUI.
*   `utius.py`: Contains utility functions for data processing, filtering, feature extraction, and plotting.
*   `requirements.txt`: Lists the project dependencies.
*   `img/`: Directory for images used in the application (e.g., logo).
*   `lacevapp.ico`, `logo.png`: Application icons and logos.
*   `LICENSE`: Contains the project's license information.
*   `README.md`: This file.

## Dependencies

The main dependencies for this project are:

*   tkinter
*   Pillow (PIL)
*   ttkthemes
*   pandas
*   numpy
*   scipy
*   matplotlib
*   seaborn
*   nolds
*   antropy

These can be installed using `pip install -r requirements.txt`.

## Contributing

Contributions are welcome! Please fork the repository, create a new branch for your features or bug fixes, and submit a pull request.

## License

This project is licensed under the terms of the LICENSE file.
