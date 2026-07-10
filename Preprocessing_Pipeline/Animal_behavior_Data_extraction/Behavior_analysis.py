import sys
import os
from pathlib import Path
import traceback
import warnings
warnings.filterwarnings('ignore')

# These imports MUST come first, before any other PyQt imports
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QCoreApplication
# Set this attribute before ANY QApplication or QCoreApplication is created
QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)

# Now import other PyQt modules
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QComboBox, QLabel, QFrame, QPushButton,
    QFileDialog, QLineEdit, QMessageBox, QTabWidget
)

# Only import these after PyQt imports
import pandas as pd
from openpyxl import load_workbook
import numpy as np
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
# import matplotlib.pyplot as plt


# ============================================================================
# TAB-DELIMITED TO CSV CONVERSION FUNCTIONS
# ============================================================================

def convert_tab_delimited_to_csv(input_folder, output_folder):
    """
    Convert .xls files that are actually tab-delimited text files to CSV format.

    Parameters:
    -----------
    input_folder : str
        Path to folder containing .xls files
    output_folder : str
        Path to folder where CSV files will be saved

    Returns:
    --------
    tuple: (converted_count, failed_files, csv_files_list)
    """

    # Convert to Path objects
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all .xls files
    xls_files = list(input_path.glob('*.xls'))

    if not xls_files:
        print(f"No .xls files found in {input_folder}")
        return 0, [], []

    print(f"Found {len(xls_files)} .xls file(s) to convert\n")

    # Convert each file
    converted_count = 0
    failed_files = []
    csv_files_list = []

    for xls_file in xls_files:
        try:
            print(f"Converting: {xls_file.name}...")

            # First, try to detect if it's a real Excel file or text file
            with open(xls_file, 'rb') as f:
                first_bytes = f.read(8)

            # Check if it's a real Excel file (starts with specific bytes)
            is_real_excel = (first_bytes[:4] == b'\xD0\xCF\x11\xE0' or
                           first_bytes[:4] == b'\x50\x4B\x03\x04')

            if is_real_excel:
                # It's a real Excel file
                df = pd.read_excel(xls_file, engine='xlrd')
                print(f"  Format: Real Excel file")
            else:
                # It's a text file with .xls extension
                try:
                    df = pd.read_csv(xls_file, sep='\t', encoding='utf-8-sig')
                    print(f"  Format: Tab-delimited text file")
                except:
                    try:
                        df = pd.read_csv(xls_file, sep=r'\s+', encoding='utf-8-sig', engine='python')
                        print(f"  Format: Space-delimited text file")
                    except:
                        df = pd.read_csv(xls_file, encoding='utf-8-sig')
                        print(f"  Format: Comma-delimited text file")

            # Save as CSV
            csv_filename = xls_file.stem + '.csv'
            csv_path = output_path / csv_filename
            df.to_csv(csv_path, index=False)

            csv_files_list.append(str(csv_path))

            print(f"  ✓ Saved: {csv_filename}")
            print(f"    Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            converted_count += 1

        except Exception as e:
            print(f"  ✗ Failed: {xls_file.name}")
            print(f"    Error: {str(e)}")
            failed_files.append((xls_file.name, str(e)))

        print()  # Empty line for readability

    # Summary
    print("="*60)
    print(f"Conversion complete!")
    print(f"Successfully converted: {converted_count}/{len(xls_files)} files")

    if failed_files:
        print(f"\nFailed files:")
        for filename, error in failed_files:
            print(f"  - {filename}")
            print(f"    {error}")

    return converted_count, failed_files, csv_files_list


class ZebraBoxGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout(self)
        
        # Add stimulus info section
        stimulus_group = QFrame()
        stimulus_group.setFrameStyle(QFrame.Panel | QFrame.Raised)
        stimulus_layout = QVBoxLayout(stimulus_group)
        
        stimulus_title = QLabel("Define Stimuli Timing")
        stimulus_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        stimulus_layout.addWidget(stimulus_title)
        
        # Add explanation label
        explanation = QLabel("Add each stimulus and its time range(s). Times should be in seconds.")
        stimulus_layout.addWidget(explanation)
        
        # Stimulus table container
        self.stimulus_container = QWidget()
        self.stimulus_container_layout = QVBoxLayout(self.stimulus_container)
        
        # Add initial stimulus entry
        self.stimuli_entries = []
        self.add_stimulus_entry()
        
        stimulus_layout.addWidget(self.stimulus_container)
        
        # Add button for more stimuli
        add_stimulus_btn = QPushButton("Add Another Stimulus")
        add_stimulus_btn.clicked.connect(self.add_stimulus_entry)
        stimulus_layout.addWidget(add_stimulus_btn)
        
        layout.addWidget(stimulus_group)
        
        # Process button
        self.process_btn = QPushButton('Process Files')
        self.process_btn.clicked.connect(self.start_processing)
        self.process_btn.setStyleSheet("font-size: 14px; padding: 10px;")
        layout.addWidget(self.process_btn)
        
        # Status label
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
    
    def add_stimulus_entry(self):
        """Add a new stimulus entry row with name and time range fields"""
        entry_widget = QWidget()
        entry_layout = QHBoxLayout(entry_widget)
        entry_layout.setContentsMargins(0, 0, 0, 0)
        
        # Stimulus name field
        name_label = QLabel("Stimulus Name:")
        stim_name = QLineEdit()
        
        # Time range fields
        start_label = QLabel("Start Time (s):")
        start_time = QLineEdit()
        end_label = QLabel("End Time (s):")
        end_time = QLineEdit()
        
        # Delete button
        delete_btn = QPushButton("Remove")
        delete_btn.clicked.connect(lambda: self.remove_stimulus_entry(entry_widget))
        
        # Add widgets to layout
        entry_layout.addWidget(name_label)
        entry_layout.addWidget(stim_name)
        entry_layout.addWidget(start_label)
        entry_layout.addWidget(start_time)
        entry_layout.addWidget(end_label)
        entry_layout.addWidget(end_time)
        entry_layout.addWidget(delete_btn)
        
        # Store the entry data
        entry_data = {
            'widget': entry_widget,
            'name': stim_name,
            'start': start_time,
            'end': end_time,
            'delete': delete_btn
        }
        
        self.stimuli_entries.append(entry_data)
        self.stimulus_container_layout.addWidget(entry_widget)
        
    def remove_stimulus_entry(self, entry_widget):
        """Remove a stimulus entry row"""
        # Find and remove the entry from the list
        for i, entry in enumerate(self.stimuli_entries):
            if entry['widget'] == entry_widget:
                self.stimuli_entries.pop(i)
                break
        
        # Remove the widget from layout and delete it
        entry_widget.setParent(None)
        entry_widget.deleteLater()
        
    def try_read_file(self, file_path):
        """Try different methods to read the input file"""
        try:
            # Try pandas' read_excel first
            df = pd.read_excel(file_path)
            self.status_label.setText(f"Reading {Path(file_path).name} as Excel")
            print(f"Column names after reading with read_excel: {df.columns.tolist()}")
            return df
        except Exception as e:
            self.status_label.setText(f"Excel read failed, trying CSV formats for {Path(file_path).name}")
            print(f"Excel read failed, trying CSV: {e}")
            try:
                # Then try read_csv with different separators
                for sep in [',', ';', '\t']:
                    try:
                        df = pd.read_csv(file_path, sep=sep)
                        self.status_label.setText(f"Read {Path(file_path).name} with separator '{sep}'")
                        print(f"Successfully read with separator '{sep}'")
                        print(f"Column names after reading with read_csv: {df.columns.tolist()}")
                        return df
                    except:
                        continue
                raise Exception(f"Failed to read file {Path(file_path).name} with any separator")
            except Exception as csv_e:
                self.status_label.setText(f"Failed to read {Path(file_path).name}")
                print(f"CSV read also failed: {csv_e}")
                raise
    
    def start_processing(self):
        # Collect stimulus information
        self.stimulus_data = self.collect_stimulus_data()
        if self.stimulus_data is None:
            return

        # Check if we have at least one stimulus defined
        if not self.stimulus_data:
            response = QMessageBox.question(
                self,
                "No Stimuli Defined",
                "No stimulus timing information was provided. Continue anyway?",
                QMessageBox.Yes | QMessageBox.No
            )
            if response == QMessageBox.No:
                return
            # Add a default "None" stimulus that covers all time
            self.stimulus_data = [{'name': 'None', 'start': 0, 'end': 99999999}]

        # Ask for input folder
        input_folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder Containing Raw Tab-Delimited Files"
        )
        if not input_folder:
            return

        # =================================================================
        # STEP 1: Convert tab-delimited files to CSV
        # =================================================================
        self.status_label.setText("Converting tab-delimited files to CSV...")
        QApplication.processEvents()

        # Create a subfolder for CSV files
        csv_folder = os.path.join(input_folder, "csv_files")

        # Convert files
        try:
            converted_count, failed_files, csv_files_list = convert_tab_delimited_to_csv(input_folder, csv_folder)

            if converted_count == 0:
                QMessageBox.warning(self, "No Files Converted",
                                  "No .xls files could be converted. Please check your input folder.")
                return

            conversion_msg = f"Converted {converted_count} files to CSV"
            if failed_files:
                conversion_msg += f"\n{len(failed_files)} files failed to convert"

            self.status_label.setText(conversion_msg)
            QApplication.processEvents()

        except Exception as e:
            QMessageBox.critical(self, "Conversion Error",
                               f"Error converting files: {str(e)}")
            return

        # =================================================================
        # STEP 2: Process the converted CSV files
        # =================================================================

        # Use the CSV folder as the processing input
        processing_folder = csv_folder

        # Get all Excel and CSV files from the CSV folder
        excel_files = [f for f in os.listdir(processing_folder) if f.endswith(('.xlsx', '.xls', '.csv'))]
        if not excel_files:
            QMessageBox.warning(self, "No Files Found",
                              "No CSV files found after conversion.")
            return
        
        # Display stimulus information
        stim_info = "Processing with the following stimuli:\n"
        for stim in self.stimulus_data:
            stim_info += f"- {stim['name']}: {stim['start']}s to {stim['end']}s\n"

        self.status_label.setText(stim_info)
        QApplication.processEvents()

        # Create an empty list to store all processed dataframes
        all_processed_dfs = []

        # Process files
        processed_count = 0
        total_files = len([f for f in excel_files if "_formatted" not in f.lower()])

        for file in excel_files:
            if "_formatted" in file.lower():
                continue

            file_path = os.path.join(processing_folder, file)
            self.status_label.setText(f"Processing {processed_count+1}/{total_files}: {file}")
            QApplication.processEvents()

            try:
                result_df = self.process_file(file_path)
                if result_df is not None:
                    all_processed_dfs.append(result_df)
                    processed_count += 1
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                print(traceback.format_exc())
                QMessageBox.warning(self, "Error", f"Error processing {file}: {str(e)}")

        if not all_processed_dfs:
            QMessageBox.warning(self, "Error", "No files could be processed successfully.")
            return

        # Combine all dataframes into one
        try:
            self.status_label.setText("Combining all processed data...")
            QApplication.processEvents()

            # Merge all dataframes by concatenating them
            combined_df = pd.concat(all_processed_dfs, ignore_index=True)

            # Sort by time
            combined_df = combined_df.sort_values(by=['Time(sec)'])

            # Automatically generate output filename with "formatted"
            output_file = os.path.join(input_folder, "formatted.xlsx")

            # Save the combined data
            self.status_label.setText("Saving combined data...")
            QApplication.processEvents()
            combined_df.to_excel(output_file, index=False)

            # Show completion message
            QMessageBox.information(self, 'Complete',
                                  f'Processing complete!\n'
                                  f'Converted {converted_count} files to CSV\n'
                                  f'Successfully processed {processed_count} files\n'
                                  f'CSV files saved to: {csv_folder}\n'
                                  f'Combined data saved to: {output_file}')
            self.status_label.setText("Ready")
        except Exception as e:
            print(f"Error combining data: {str(e)}")
            print(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Error combining data: {str(e)}")

    def collect_stimulus_data(self):
        """Collect all stimulus information from the entry fields"""
        stimulus_data = []
        
        for entry in self.stimuli_entries:
            name = entry['name'].text().strip()
            start_text = entry['start'].text().strip()
            end_text = entry['end'].text().strip()
            
            # Skip empty entries
            if not name or not start_text or not end_text:
                continue
                
            try:
                start_time = float(start_text)
                end_time = float(end_text)
                
                if start_time >= end_time:
                    QMessageBox.warning(self, "Invalid Time Range", 
                                      f"Stimulus '{name}': End time must be greater than start time.")
                    return None
                
                stimulus_data.append({
                    'name': name,
                    'start': start_time,
                    'end': end_time
                })
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", 
                                  f"Stimulus '{name}': Start and end times must be numbers.")
                return None
                
        return stimulus_data
    
    def get_stimulus_name(self, time_sec, stimulus_data):
        """Find which stimulus applies at a given time point"""
        for stim in stimulus_data:
            if stim['start'] <= time_sec <= stim['end']:
                return stim['name']
        return "None"  # Default when no stimulus is active
        
    def process_file(self, file_path):
        """Process a single file and return a dataframe with the results"""
        try:
            print(f"Attempting to read file: {file_path}")
            # Read the file
            df = self.try_read_file(file_path)
            
            # Print available columns to debug
            print(f"Available columns: {df.columns.tolist()}")
            
            # Handle case sensitivity by converting all column names to lowercase
            df.columns = [col.lower().strip() for col in df.columns]
            
            # Check if required columns exist
            required_columns = ["location", "data1", "time"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"Missing required columns: {missing_columns}")
                print(f"Available columns (lowercase): {df.columns.tolist()}")
                raise ValueError(f"Required columns missing: {missing_columns}")
                
            # Filter only needed columns
            filtered_df = df[["time", "location", "data1"]]
            
            # Filter out rows where data1 is not numeric
            print("Checking for non-numeric values in data1 column...")
            filtered_df = filtered_df.copy()  # Create a copy to avoid SettingWithCopyWarning
            
            # Try to convert data1 to numeric, coercing errors to NaN
            filtered_df['data1'] = pd.to_numeric(filtered_df['data1'], errors='coerce')
            
            # Count how many non-numeric rows were found
            non_numeric_count = filtered_df['data1'].isna().sum()
            if non_numeric_count > 0:
                print(f"Found {non_numeric_count} rows with non-numeric data1 values, removing them")
                # Remove rows where data1 is NaN (was non-numeric)
                filtered_df = filtered_df.dropna(subset=['data1'])
            
            # Get unique locations (in the original order they appear)
            unique_locations = pd.unique(filtered_df['location'])
            print(f"Found {len(unique_locations)} unique locations")
            
            # Create result DataFrame
            all_locations = [f'Loc{str(i).zfill(2)}' for i in range(1, 97)]  # All possible locations
            
            # Get all unique time points and convert to seconds
            unique_times = filtered_df['time'].unique()
            time_seconds = unique_times / 1000000.0  # Convert microseconds to seconds
            
            # Create result DataFrame with time points and stimulus column
            result_df = pd.DataFrame(index=range(len(time_seconds)), columns=['Time(sec)', 'Stimuli'] + all_locations)
            result_df['Time(sec)'] = time_seconds
            
            # Get stimulus information 
            stimulus_data = self.stimulus_data
            
            # Add stimulus information for each time point
            for idx, time_sec in enumerate(time_seconds):
                result_df.at[idx, 'Stimuli'] = self.get_stimulus_name(time_sec, stimulus_data)
            
            # Fill dataframe with data1 values at the corresponding time and location
            for _, row in filtered_df.iterrows():
                time_sec = row['time'] / 1000000.0
                loc = row['location']
                if loc in all_locations:
                    # Find the row index where Time(sec) equals time_sec
                    row_idx = result_df.index[result_df['Time(sec)'] == time_sec].tolist()
                    if row_idx:
                        result_df.at[row_idx[0], loc] = row['data1']
            
            # Fill missing values with NaN
            result_df = result_df.fillna(np.nan)
            
            print(f"\nProcessed {Path(file_path).name}")
            return result_df
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            print(traceback.format_exc())
            return None

class BehaviorAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Behavior Analysis Tool")
        self.setup_ui()
    
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create and add Raw Data Processing tab
        self.raw_processing_tab = ZebraBoxGUI()
        self.tab_widget.addTab(self.raw_processing_tab, "Raw Data Processing")
        
        # Create and add Behavior Analysis tab
        self.behavior_analysis_tab = QWidget()
        behavior_layout = QVBoxLayout(self.behavior_analysis_tab)
        
        upload_btn = QPushButton("Upload Excel File")
        upload_btn.clicked.connect(self.upload_file)
        behavior_layout.addWidget(upload_btn)
        
        # Add a status label to show processing status
        self.status_label = QLabel("Ready")
        behavior_layout.addWidget(self.status_label)
        
        self.tab_widget.addTab(self.behavior_analysis_tab, "Behavior Analysis")
        
        layout.addWidget(self.tab_widget)

    def upload_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Excel File",
            "",
            "Excel Files (*.xlsx)"
        )
        if file_path:
            self.status_label.setText("Processing data... This may take a moment.")
            QApplication.processEvents()  # Update the UI
            
            try:
                result = group_matching_columns(file_path)
                self.status_label.setText("Data processed successfully. Showing visualization.")
                self.show_visualization(result)
            except Exception as e:
                error_msg = f"Error processing file: {str(e)}"
                self.status_label.setText(error_msg)
                QMessageBox.critical(self, "Processing Error", error_msg)

    def show_visualization(self, result):
        self.vis_window = GeneVisualizationWindow(
            df=result['dataframe'],
            group_averages=result['group_averages'], 
            stimulus_col='Stimuli',
            error_data=result['error_data'],
            whole_average_sem=result['whole_average_sem']
        )
        self.vis_window.show()
    
   
    def closeEvent(self, event):
        if hasattr(self, 'vis_window'):
            self.vis_window.close()
        event.accept()

def calculate_sem(df, group_columns):
    """Calculate Standard Error of Mean for a group of columns"""
    numeric_data = df[group_columns].apply(pd.to_numeric, errors='coerce')
    sem = numeric_data.sem(axis=1)
    return sem

def calculate_whole_average_sem(df, all_numeric_columns):
    """Calculate Standard Error of Mean for all numeric columns"""
    numeric_data = df[all_numeric_columns].apply(pd.to_numeric, errors='coerce')
    sem = numeric_data.sem(axis=1)
    return sem


class GeneVisualizationWindow(QMainWindow):
    def __init__(self, df, group_averages, stimulus_col, error_data, whole_average_sem):
        super().__init__()
        self.df = df
        # Filter out WT_Average from group_averages if present
        self.group_averages = [avg for avg in (group_averages or []) if avg != 'WT_Average']
        self.stimulus_col = stimulus_col
        self.error_data = error_data
        self.whole_average_sem = whole_average_sem
        self.wt_only_mode = len(self.group_averages) == 0
        
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle("Behavior Analysis Visualization")
        self.setGeometry(100, 100, 1200, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create control panel
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)
        
        # Gene Group Selection (only show if there are gene groups)
        if not self.wt_only_mode:
            group_label = QLabel("Select Gene Group:")
            self.group_combo = QComboBox()
            self.group_combo.addItems(sorted(self.group_averages))
            self.group_combo.currentTextChanged.connect(self.update_plot)
            control_layout.addWidget(group_label)
            control_layout.addWidget(self.group_combo)
        
        # Stimulus Selection
        stim_label = QLabel("Select Stimulus:")
        self.stim_combo = QComboBox()
        stimuli = sorted(self.df[self.stimulus_col].unique())
        self.stim_combo.addItems(stimuli + ["All Stimuli (Combined)"])
        self.stim_combo.currentTextChanged.connect(self.update_plot)
        
        # Color Selection
        color_label = QLabel("WT Line Color:")
        self.color_combo = QComboBox()
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'black', 'brown', 'gray']
        self.color_combo.addItems(colors)
        self.color_combo.setCurrentText('red')
        self.color_combo.currentTextChanged.connect(self.update_plot)
        
        # Add export button
        self.export_btn = QPushButton("Export Plot")
        self.export_btn.clicked.connect(self.export_plot)
        
        # Add widgets to control layout
        for widget in [stim_label, self.stim_combo,
                      color_label, self.color_combo,
                      self.export_btn]:
            control_layout.addWidget(widget)
        
        control_layout.addStretch()
        
        # Create matplotlib figure
        self.figure = plt.figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        
        # Add the matplotlib toolbar (this adds zoom, pan, save, etc.)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # Add the toolbar and canvas to the layout
        main_layout.addWidget(control_panel)
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas)
        
        # Initial plot
        self.update_plot()
    
    def export_plot(self):
        """Export the current plot as a high-resolution image"""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Export Plot",
            "",
            "PNG Files (*.png);;JPEG Files (*.jpg);;PDF Files (*.pdf);;SVG Files (*.svg)",
            options=options
        )
        
        if file_name:
            # Add file extension if not present
            if not any(file_name.endswith(ext) for ext in ['.png', '.jpg', '.pdf', '.svg']):
                file_name += '.png'
                
            # Export at high resolution (600 DPI)
            self.figure.savefig(file_name, dpi=600, bbox_inches='tight')
            QMessageBox.information(self, "Export Complete", f"Plot exported to {file_name}")
            
    def update_plot(self):
        if self.df is None or self.stimulus_col not in self.df.columns:
            return
            
        selected_stim = self.stim_combo.currentText()
        selected_gene = None if self.wt_only_mode else self.group_combo.currentText()
        wt_color = self.color_combo.currentText()
        
        self.figure.clear()
    
        try:
            # Make sure all stimulus values are converted to strings for comparison
            if self.stimulus_col in self.df.columns:
                self.df[self.stimulus_col] = self.df[self.stimulus_col].astype(str)
            
            if selected_stim == "All Stimuli (Combined)":
                # Create single plot for all stimuli combined
                ax = self.figure.add_subplot(111)
                
                # Enable grid for easier reading when zooming
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # Use actual time values instead of indices
                time_points = self.df['Time(sec)'].values if 'Time(sec)' in self.df.columns else np.arange(len(self.df))
                
                # Plot WT average
                if 'WT_Average' in self.df.columns:
                    ax.plot(time_points, 
                           self.df['WT_Average'],
                           label='WT Average',
                           color=wt_color,
                           alpha=0.7,
                           linewidth=1)
                
                # Plot selected gene group if not in WT-only mode
                if not self.wt_only_mode and selected_gene and selected_gene in self.df.columns:
                    ax.plot(time_points, 
                           self.df[selected_gene],
                           label=selected_gene,
                           linewidth=1)
                
                # Adjust plot for stimulus labels
                plt.subplots_adjust(top=0.85)
                
                # Add stimulus labels
                prev_stim = None
                start_idx = 0
                
                # Make sure we have valid y-limits first
                if ax.get_ylim()[1] > ax.get_ylim()[0]:
                    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                    
                    for idx, stim in enumerate(self.df[self.stimulus_col]):
                        if stim != prev_stim:
                            if prev_stim is not None:
                                if idx > start_idx:  # Ensure we have a valid range
                                    mid_point_idx = start_idx + (idx - start_idx) // 2
                                    mid_point_time = time_points[mid_point_idx]
                                    ax.text(mid_point_time, ax.get_ylim()[1] + (y_range * 0.05), str(prev_stim),
                                           rotation=90, ha='center', va='bottom')
                                    ax.axvline(x=time_points[idx], color='gray', linestyle='--', alpha=0.3)
                            start_idx = idx
                            prev_stim = stim
                    
                    # Handle last stimulus
                    if len(self.df) > start_idx:  # Ensure we have data left
                        mid_point_idx = start_idx + (len(self.df) - start_idx) // 2
                        if mid_point_idx < len(time_points):  # Check if index is valid
                            mid_point_time = time_points[mid_point_idx]
                            ax.text(mid_point_time, ax.get_ylim()[1] + (y_range * 0.05), str(prev_stim),
                                   rotation=90, ha='center', va='bottom')
                
                # Set x-axis ticks to multiples of 10
                if len(time_points) > 0:
                    max_time = max(time_points)
                    # Calculate tick interval (ensure it's a multiple of 10)
                    tick_interval = max(10, int(max_time // 20) // 10 * 10) 
                    
                    # Generate tick positions as multiples of 10
                    tick_positions = np.arange(0, max_time + tick_interval, tick_interval)
                    ax.set_xticks(tick_positions)
                    ax.set_xticklabels([f"{int(t)}" for t in tick_positions], rotation=45)
                
                ax.set_xlabel('Time (sec)', fontsize=10)
                ax.set_ylabel('Value', fontsize=10)
                ax.tick_params(axis='both', labelsize=10)
                
                # Create a single legend with unique entries
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), fontsize=10)
        
            else:
                # Single stimulus plot
                mask = self.df[self.stimulus_col] == selected_stim
                if not any(mask):
                    # If no data matches, show an empty plot with a message
                    ax = self.figure.add_subplot(111)
                    ax.text(0.5, 0.5, f"No data found for stimulus: {selected_stim}",
                           horizontalalignment='center', verticalalignment='center',
                           transform=ax.transAxes, fontsize=14)
                else:
                    stim_data = self.df[mask]
                    
                    # Use actual time values
                    x_points = stim_data['Time(sec)'].values if 'Time(sec)' in stim_data.columns else np.arange(len(stim_data))
                    
                    ax = self.figure.add_subplot(111)
                    
                    # Enable grid for easier reading when zooming
                    ax.grid(True, linestyle='--', alpha=0.7)
                    
                    # Plot WT average with error bars if it exists
                    if 'WT_Average' in stim_data.columns:
                        # Safely handle error data to prevent "too many values to unpack" error
                        try:
                            # Get WT error data and make sure it has the same length as the filtered data
                            wt_errors = self.error_data.get('WT', np.zeros(len(self.df)))
                            
                            # Check if the error data matches the mask
                            if len(wt_errors) == len(self.df):
                                filtered_wt_errors = wt_errors[mask]
                                
                                # Make sure error data matches the length of x_points
                                if len(filtered_wt_errors) == len(x_points):
                                    ax.errorbar(x_points, 
                                               stim_data['WT_Average'],
                                               yerr=filtered_wt_errors,
                                               label='WT Average',
                                               marker='o',
                                               color=wt_color,
                                               alpha=0.7,
                                               linewidth=1,
                                               markersize=4,
                                               capsize=3)
                                else:
                                    # If lengths don't match, plot without error bars
                                    print(f"Warning: Error data length ({len(filtered_wt_errors)}) doesn't match data length ({len(x_points)}), plotting without error bars")
                                    ax.plot(x_points, 
                                           stim_data['WT_Average'],
                                           label='WT Average',
                                           marker='o',
                                           color=wt_color,
                                           alpha=0.7,
                                           linewidth=1,
                                           markersize=4)
                            else:
                                # If error data length doesn't match original data, plot without error bars
                                print(f"Warning: Error data length mismatch, plotting without error bars")
                                ax.plot(x_points, 
                                       stim_data['WT_Average'],
                                       label='WT Average',
                                       marker='o',
                                       color=wt_color,
                                       alpha=0.7,
                                       linewidth=1,
                                       markersize=4)
                        except Exception as err:
                            print(f"Error plotting WT with error bars: {str(err)}")
                            # Fallback to simple plotting without error bars
                            ax.plot(x_points, 
                                   stim_data['WT_Average'],
                                   label='WT Average',
                                   marker='o',
                                   color=wt_color,
                                   alpha=0.7,
                                   linewidth=1,
                                   markersize=4)
                    
                    # Plot selected gene group if not in WT-only mode
                    if not self.wt_only_mode and selected_gene and selected_gene in stim_data.columns:
                        try:
                            # Similar safe error handling for gene data
                            gene_base = selected_gene.replace('_Average', '')
                            gene_errors = self.error_data.get(gene_base, np.zeros(len(self.df)))
                            
                            if len(gene_errors) == len(self.df):
                                filtered_gene_errors = gene_errors[mask]
                                
                                if len(filtered_gene_errors) == len(x_points):
                                    ax.errorbar(x_points, 
                                               stim_data[selected_gene],
                                               yerr=filtered_gene_errors,
                                               label=selected_gene,
                                               marker='o',
                                               linewidth=1,
                                               markersize=4,
                                               capsize=3)
                                else:
                                    # Plot without error bars if lengths don't match
                                    ax.plot(x_points, 
                                           stim_data[selected_gene],
                                           label=selected_gene,
                                           marker='o',
                                           linewidth=1,
                                           markersize=4)
                            else:
                                # Plot without error bars
                                ax.plot(x_points, 
                                       stim_data[selected_gene],
                                       label=selected_gene,
                                       marker='o',
                                       linewidth=1,
                                       markersize=4)
                        except Exception as err:
                            print(f"Error plotting gene with error bars: {str(err)}")
                            # Fallback to simple plotting
                            ax.plot(x_points, 
                                   stim_data[selected_gene],
                                   label=selected_gene,
                                   marker='o',
                                   linewidth=1,
                                   markersize=4)
                    
                    # Set x-axis ticks to multiples of 10
                    if len(x_points) > 0:
                        min_time = min(x_points)
                        max_time = max(x_points)
                        
                        # Calculate tick interval (ensure it's a multiple of 10)
                        # For shorter ranges, use smaller intervals
                        time_range = max_time - min_time
                        if time_range <= 50:
                            tick_interval = 10  # For small ranges, use 10-second intervals
                        else:
                            tick_interval = max(10, int(time_range // 15) // 10 * 10)  # Ensure it's a multiple of 10
                        
                        # Calculate start point (make sure it's a multiple of 10)
                        start_tick = int(min_time // 10) * 10
                        
                        # Generate tick positions as multiples of 10
                        tick_positions = np.arange(
                            start_tick,
                            max_time + tick_interval,
                            tick_interval
                        )
                        ax.set_xticks(tick_positions)
                        ax.set_xticklabels([f"{int(t)}" for t in tick_positions], rotation=45)
                    
                    ax.set_title(f'Response to {selected_stim}', fontsize=12)
                    ax.set_xlabel('Time (sec)', fontsize=10)
                    ax.set_ylabel('Value', fontsize=10)
                    ax.tick_params(axis='both', labelsize=10)
                    
                    # Create a single legend with unique entries
                    handles, labels = ax.get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    ax.legend(by_label.values(), by_label.keys(), fontsize=10)
            
            self.figure.tight_layout()
            self.canvas.draw()
        
        except Exception as e:
            # Show error in the plot
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            error_msg = f"Error creating plot: {str(e)}"
            ax.text(0.5, 0.5, error_msg, ha='center', va='center', transform=ax.transAxes)
            print(f"Visualization error: {str(e)}")
            print(traceback.format_exc())
            self.canvas.draw()
            
    def add_cursor_coordinates(self):
        """Add a status bar that shows cursor coordinates when hovering over the plot"""
        self.statusbar = self.statusBar()
        
        def format_coord(x, y):
            self.statusbar.showMessage(f"Time: {x:.2f} sec, Value: {y:.2f}")
            return ''
        
        for ax in self.figure.get_axes():
            ax.format_coord = format_coord
            
    def add_data_cursor(self):
        """Add a data cursor that shows the nearest data point when clicking on the plot"""
        from matplotlib.widgets import Cursor
        
        for ax in self.figure.get_axes():
            # Add crosshair cursor
            cursor = Cursor(ax, useblit=True, color='red', linewidth=0.5)
            
            # Connected event for clicking on data points
            def on_click(event):
                if event.inaxes:
                    # Find the nearest data point
                    ax = event.inaxes
                    lines = ax.get_lines()
                    
                    if lines:
                        min_dist = float('inf')
                        nearest_point = None
                        nearest_line_label = None
                        
                        for line in lines:
                            data = line.get_data()
                            x_data, y_data = data[0], data[1]
                            
                            for i in range(len(x_data)):
                                dist = ((x_data[i] - event.xdata)**2 + 
                                       (y_data[i] - event.ydata)**2)**0.5
                                
                                if dist < min_dist:
                                    min_dist = dist
                                    nearest_point = (x_data[i], y_data[i])
                                    nearest_line_label = line.get_label()
                        
                        if nearest_point and min_dist < 5:  # Threshold to be close enough
                            # Show annotation with data point info
                            self.statusbar.showMessage(
                                f"{nearest_line_label}: Time = {nearest_point[0]:.2f}, Value = {nearest_point[1]:.2f}"
                            )
            
            # Connect the event
            self.canvas.mpl_connect('button_press_event', on_click)
        
    def closeEvent(self, event):
        plt.close('all')
        event.accept()
        

def get_base_column_name(column_name):
    """
    Get the base name of the column handling both cases:
    1. Names with .digit at the end (e.g., 'gene.1', 'gene.2')
    2. Exact same names without digits (e.g., 'gene', 'gene')
    """
    # First, try splitting by dot for .digit case
    parts = column_name.split('.')
    if len(parts) > 1 and parts[-1].isdigit():
        return '.'.join(parts[:-1])
    # If no dot or not a digit after dot, return the full name for exact matching
    return column_name

def group_matching_columns(file_path):
    try:
        print(f"Reading file: {file_path}")
        
        # Try multiple methods to read the file
        try:
            df = pd.read_excel(file_path)
        except Exception as excel_error:
            print(f"Error reading Excel file: {str(excel_error)}")
            try:
                df = pd.read_csv(file_path)
                print("Successfully read the file as CSV")
            except Exception as csv_error:
                print(f"All reading methods failed. Last error: {str(csv_error)}")
                raise Exception("Unable to read the file. The file may be corrupted.")
        
        print("Successfully read the file")
        print("Columns in file:", df.columns.tolist())
        
        # Print first few rows to diagnose data issues
        print("\nFirst few rows of data:")
        print(df.head().to_string())
        
        # Step 1: Force explicit conversion of all numeric columns
        excluded_columns = ['Bin', 'Stimuli']  # Columns that should remain as strings
        
        # Force numeric conversion for all data columns
        for col in df.columns:
            if col not in excluded_columns and col != 'Time(sec)':
                # Convert column to numeric, ignoring errors (which sets problem values to NaN)
                try:
                    # First, try direct conversion
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    print(f"Converted {col} to numeric - direct method")
                except Exception as e:
                    # If that fails, try more aggressive cleanup
                    try:
                        # Convert to string first to handle mixed types
                        str_series = df[col].astype(str)
                        # Clean the string
                        cleaned = str_series.str.replace(',', '.').str.replace(' ', '').str.replace('%', '')
                        # Convert back to numeric
                        df[col] = pd.to_numeric(cleaned, errors='coerce')
                        print(f"Converted {col} to numeric - cleanup method")
                    except Exception as e2:
                        print(f"Failed to convert {col}: {str(e2)}")
                
                # Check if conversion was successful
                non_na_count = df[col].count()
                total_count = len(df[col])
                if non_na_count == 0:
                    print(f"WARNING: Column {col} has all NaN values after conversion!")
                else:
                    print(f"Column {col}: {non_na_count}/{total_count} values valid ({non_na_count/total_count*100:.1f}%)")
                    print(f"  Range: {df[col].min()} to {df[col].max()}")
        
        # Make sure Time(sec) is numeric
        if 'Time(sec)' in df.columns:
            df['Time(sec)'] = pd.to_numeric(df['Time(sec)'], errors='coerce')
            print(f"Converted Time(sec) to numeric: {df['Time(sec)'].dtype}")
        
        # Ensure Stimuli column is string type
        if 'Stimuli' in df.columns:
            df['Stimuli'] = df['Stimuli'].astype(str)
            print(f"Converted Stimuli to string: {df['Stimuli'].dtype}")
        
        # First, get a list of all column names and their counts
        col_counts = df.columns.value_counts()
        print("\nColumn name counts:")
        for col, count in col_counts.items():
            print(f"{col}: {count} occurrences")
        
        column_groups = {}
        excluded_columns = ['Bin', 'Stimuli', 'Time(sec)']
        all_numeric_columns = []
        error_data = {}
        group_averages = []
        
        # Group columns by exact name or base name
        for col in df.columns:
            if col in excluded_columns:
                continue
            
            base_name = get_base_column_name(col)
            
            # Check for exact matches first
            exact_matches = [c for c in df.columns if c == col]
            if len(exact_matches) > 1:
                base_name = col
            
            if base_name not in column_groups:
                column_groups[base_name] = []
            if col not in column_groups[base_name]:
                column_groups[base_name].append(col)
            if col not in all_numeric_columns:
                all_numeric_columns.append(col)
        
        # Print grouping information
        print("\nIdentified groups:")
        for base_name, columns in column_groups.items():
            print(f"\nBase name: {base_name}")
            print(f"Grouped columns: {columns}")
            print(f"Count: {len(columns)}")
        
        new_columns = []
        grouped_data = {}
        
        # Add excluded columns first
        for col in excluded_columns:
            if col in df.columns:
                new_columns.append(col)
                grouped_data[col] = df[col].copy()
        
        # Process each group - SIMPLIFIED TO AVOID CONVERSION ISSUES
        for base_name, columns in sorted(column_groups.items()):
            print(f"\nProcessing group: {base_name}")
            print(f"Found columns: {columns}")
            
            # Add original columns - ALREADY CONVERTED ABOVE
            for col in sorted(columns):
                new_columns.append(col)
                grouped_data[col] = df[col]  # Data already converted to numeric above
            
            # Calculate average and SEM for groups with multiple columns
            if len(columns) > 1:
                avg_col_name = f"{base_name}_Average"
                new_columns.append(avg_col_name)
                
                # Create a temporary dataframe for this group with only numeric data
                group_df = df[columns].copy()
                
                # Ensure all columns are numeric and calculate mean
                avg_values = group_df.mean(axis=1)
                grouped_data[avg_col_name] = avg_values
                
                # Only add non-WT averages to group_averages
                if base_name != 'WT':
                    group_averages.append(avg_col_name)
                
                # Calculate and store SEM
                error_data[base_name] = group_df.sem(axis=1)
                print(f"Added average column: {avg_col_name}")
                
                # Check if average column has valid values
                non_na_count = avg_values.count()
                total_count = len(avg_values)
                print(f"Average column {avg_col_name}: {non_na_count}/{total_count} valid values ({non_na_count/total_count*100:.1f}%)")
                if non_na_count > 0:
                    print(f"  Range: {avg_values.min()} to {avg_values.max()}")
                else:
                    print("  WARNING: No valid values in average column!")
                    
            elif base_name == 'WT':
                # Handle single WT column case
                avg_col_name = "WT_Average"
                new_columns.append(avg_col_name)
                grouped_data[avg_col_name] = df[columns[0]]  # Already numeric from above
                error_data['WT'] = np.zeros(len(df))  # No error bars for single column
                
                # Check if column has valid values
                non_na_count = df[columns[0]].count()
                total_count = len(df[columns[0]])
                print(f"WT_Average column: {non_na_count}/{total_count} valid values ({non_na_count/total_count*100:.1f}%)")
                if non_na_count > 0:
                    print(f"  Range: {df[columns[0]].min()} to {df[columns[0]].max()}")
                else:
                    print("  WARNING: No valid values in WT_Average column!")
        
        # Calculate whole average SEM
        numeric_df = df[all_numeric_columns].copy()
        whole_average_sem = numeric_df.sem(axis=1)

        # Create final DataFrame
        df_grouped = pd.DataFrame(grouped_data)
        
        # Ensure we only include columns that actually exist
        valid_columns = [col for col in new_columns if col in df_grouped.columns]
        df_grouped = df_grouped[valid_columns]
        
        # Save to a new file
        base_filepath = file_path.rsplit('.', 1)[0]
        output_path = f"{base_filepath}_analyzed.xlsx"
        
        # Save the data to a new file
        print(f"Saving data to new file: {output_path}")
        
        try:
            df_grouped.to_excel(output_path, sheet_name='Grouped_Data', index=False)
        except Exception as write_error:
            print(f"Error saving to Excel: {str(write_error)}")
            # Fallback to CSV if Excel save fails
            csv_path = f"{base_filepath}_analyzed.xls"
            print(f"Attempting to save as CSV to: {csv_path}")
            df_grouped.to_csv(csv_path, index=False)
            output_path = csv_path
        
        print(f"\nSaved to file: '{output_path}'")
        
        # Check if we have any numeric data at all
        has_numeric_data = False
        for col in df_grouped.columns:
            if col not in excluded_columns:
                if df_grouped[col].count() > 0:
                    has_numeric_data = True
                    break
        
        if not has_numeric_data:
            print("\n*** WARNING: No valid numeric data found in any column! ***")
            print("Please check your input file for data formatting issues.")
        
        return {
            'dataframe': df_grouped,
            'group_averages': group_averages,
            'error_data': error_data,
            'whole_average_sem': whole_average_sem
        }
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print(traceback.format_exc())
        raise e

def main():
    # QApplication already has Qt.AA_ShareOpenGLContexts set above
    app = QApplication(sys.argv)
    window = BehaviorAnalyzer()
    window.setGeometry(100, 100, 1200, 800)
    window.show()
    sys.exit(app.exec_())
if __name__ == "__main__":
    main()