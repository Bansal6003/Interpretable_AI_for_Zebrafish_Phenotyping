import sys
import pandas as pd
import os
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                             QWidget, QLabel, QFileDialog, QTextEdit, QProgressBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import re

def natural_sort_key(path):
    """
    Sort files naturally (1, 2, 3... 10, 11, 12 instead of 1, 10, 11... 2, 3)
    """
    filename = path.stem  # Get filename without extension
    # Extract numbers from filename
    numbers = re.findall(r'\d+', filename)
    if numbers:
        return int(numbers[0])
    return 0

class ExcelCombinerThread(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, folder_path, output_path):
        super().__init__()
        self.folder_path = folder_path
        self.output_path = output_path
    
    def run(self):
        try:
            # Get all Excel files
            excel_files = list(Path(self.folder_path).glob("*.xlsx"))
            excel_files.extend(list(Path(self.folder_path).glob("*.xls")))
            
            if not excel_files:
                self.error.emit("No Excel files found in the selected folder!")
                return
            
            # Sort files naturally (1, 2, 3... 10, 11, 12)
            excel_files.sort(key=natural_sort_key)
            
            self.progress.emit(5, f"Found {len(excel_files)} Excel files")
            self.progress.emit(7, f"File order: {', '.join([f.name for f in excel_files])}")
            
            # Read first file for Time and Stimuli columns
            first_df = pd.read_excel(excel_files[0])
            
            if 'Time(sec)' not in first_df.columns or 'Stimuli' not in first_df.columns:
                self.error.emit("First file doesn't have 'Time(sec)' and 'Stimuli' columns!")
                return
            
            # Get maximum row count across all files
            self.progress.emit(10, "Checking row counts in all files...")
            max_rows = 0
            for file in excel_files:
                df = pd.read_excel(file)
                max_rows = max(max_rows, len(df))
            self.progress.emit(12, f"Maximum rows across all files: {max_rows}")
            
            # Start with Time and Stimuli from first file, padded to max_rows
            time_col = list(first_df['Time(sec)']) + [None] * (max_rows - len(first_df))
            stimuli_col = list(first_df['Stimuli']) + [None] * (max_rows - len(first_df))
            
            # List to collect all dataframes for concatenation
            all_dataframes = [pd.DataFrame({
                'Time(sec)': time_col,
                'Stimuli': stimuli_col
            })]
            
            # Separate list for mutants only (no Time/Stimuli, no Loc##)
            mutant_dataframes = []
            
            self.progress.emit(15, f"Initialized with Time(sec) and Stimuli from {excel_files[0].name}")
            
            # Track all columns being added
            total_behavior_columns = 0
            total_mutant_columns = 0
            all_mutant_names = []  # Track all mutant column names
            
            # Process each file
            total_files = len(excel_files)
            for idx, file in enumerate(excel_files):
                self.progress.emit(15 + int((idx / total_files) * 80), f"Processing: {file.name}")
                
                df = pd.read_excel(file)
                
                # Get behavior columns (all except Time(sec) and Stimuli)
                behavior_cols = [col for col in df.columns if col not in ['Time(sec)', 'Stimuli']]

                # Separate mutants from unlabeled wells (Loc##)
                # Mutants are anything that's NOT in the pattern Loc## (like Loc01, Loc02, etc.)
                mutant_cols = [col for col in behavior_cols if not re.match(r'^Loc\d+$', col)]
                
                self.progress.emit(15 + int((idx / total_files) * 80), 
                                 f"  → Found {len(behavior_cols)} behavior columns ({len(mutant_cols)} mutants)")
                
                # Create a dictionary for this file's columns (keep original column names)
                file_data = {}
                mutant_data = {}
                
                for col in behavior_cols:
                    # Pad column data to max_rows if needed
                    col_data = list(df[col]) + [None] * (max_rows - len(df))
                    
                    # Add to all columns
                    file_data[col] = col_data
                    total_behavior_columns += 1
                    
                    # If it's a mutant, also add to mutant-only collection
                    if col in mutant_cols:
                        all_mutant_names.append(col)
                        mutant_data[col] = col_data
                        total_mutant_columns += 1
                
                # Create dataframe for this file's columns and add to list
                all_dataframes.append(pd.DataFrame(file_data))
                
                # Add mutant dataframe if there are any mutants
                if mutant_data:
                    mutant_dataframes.append(pd.DataFrame(mutant_data))
                
                self.progress.emit(15 + int((idx / total_files) * 80), 
                                 f"  ✓ Added {len(behavior_cols)} columns from {file.name} (Total: {total_behavior_columns})")
            
            # Concatenate all dataframes at once (MUCH faster!)
            self.progress.emit(90, "Concatenating all columns together...")
            combined_df = pd.concat(all_dataframes, axis=1)

            # Remove pandas' auto-generated suffixes (.1, .2, .3, etc.) from duplicate column names
            self.progress.emit(92, "Removing pandas auto-generated suffixes from column names...")
            cleaned_columns = []
            for col in combined_df.columns:
                # Remove .1, .2, .3 etc suffixes that pandas adds to duplicates
                cleaned_col = re.sub(r'\.\d+$', '', str(col))
                cleaned_columns.append(cleaned_col)
            combined_df.columns = cleaned_columns

            # Save combined file (all columns) as CSV to preserve duplicate column names
            self.progress.emit(95, "Saving combined file (all columns) as CSV...")
            # Change extension to .csv
            csv_output_path = self.output_path.rsplit('.', 1)[0] + '.csv'
            combined_df.to_csv(csv_output_path, index=False)

            summary = (f"Successfully combined {total_files} files!\n\n"
                      f"Output (CSV): {csv_output_path}\n"
                      f"Total columns: {len(combined_df.columns)}\n"
                      f"  - Time(sec) + Stimuli: 2 columns\n"
                      f"  - Behavior data: {total_behavior_columns} columns\n"
                      f"Total rows: {len(combined_df)}\n\n"
                      f"NOTE: Saved as CSV to preserve duplicate column names without .1, .2 suffixes.\n"
                      f"You can open CSV files in Excel.\n\n"
                      f"Use the 'Remove Loc## Columns' button to create a mutants-only file.")
            
            self.progress.emit(100, "Complete!")
            self.finished.emit(summary)
            
        except Exception as e:
            import traceback
            self.error.emit(f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}")


class SuffixRemoverThread(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, input_path, output_path):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path

    def run(self):
        try:
            self.progress.emit(10, f"Reading file: {self.input_path}")

            # Read the file (supports both CSV and Excel)
            if self.input_path.endswith('.csv'):
                df = pd.read_csv(self.input_path)
            else:
                df = pd.read_excel(self.input_path)

            self.progress.emit(30, f"Found {len(df.columns)} columns")
            self.progress.emit(40, "Removing .1, .2, .3 suffixes...")

            # Remove suffixes but preserve "nkx2.7" as it's an original name
            cleaned_columns = []
            suffix_count = 0

            for col in df.columns:
                col_str = str(col)

                # Special case: if this is exactly "nkx2.7", keep it
                if col_str == "nkx2.7":
                    cleaned_columns.append(col_str)
                else:
                    # Remove last .N suffix
                    cleaned_col = re.sub(r'\.\d+$', '', col_str)
                    if cleaned_col != col_str:
                        suffix_count += 1
                    cleaned_columns.append(cleaned_col)

            df.columns = cleaned_columns

            self.progress.emit(70, f"Removed suffixes from {suffix_count} columns")

            # Save as CSV to preserve duplicate column names
            self.progress.emit(80, f"Saving to: {self.output_path}")
            csv_output_path = self.output_path.rsplit('.', 1)[0] + '.csv'
            df.to_csv(csv_output_path, index=False)

            summary = (f"Successfully removed suffixes!\n\n"
                      f"Input: {self.input_path}\n"
                      f"Output (CSV): {csv_output_path}\n\n"
                      f"Removed suffixes from {suffix_count} columns\n"
                      f"Total columns: {len(df.columns)}\n"
                      f"Total rows: {len(df)}\n\n"
                      f"NOTE: 'nkx2.7' preserved as original name.")

            self.progress.emit(100, "Complete!")
            self.finished.emit(summary)

        except Exception as e:
            import traceback
            self.error.emit(f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}")


class LocRemoverThread(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, input_path, output_path):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path

    def run(self):
        try:
            self.progress.emit(10, f"Reading file: {self.input_path}")

            # Read the combined file (supports both CSV and Excel)
            if self.input_path.endswith('.csv'):
                df = pd.read_csv(self.input_path)
            else:
                df = pd.read_excel(self.input_path)

            self.progress.emit(30, f"Found {len(df.columns)} columns")

            # Filter out all Loc## columns (with or without suffixes like Loc01.1)
            filtered_columns = [col for col in df.columns
                              if not re.match(r'^Loc\d+(\.\d+)*$', col)]

            loc_count = len(df.columns) - len(filtered_columns)

            self.progress.emit(50, f"Removing {loc_count} Loc## columns...")

            # Create new dataframe with filtered columns
            mutants_df = df[filtered_columns]

            # Save as CSV to preserve duplicate column names
            self.progress.emit(80, f"Saving to: {self.output_path}")
            csv_output_path = self.output_path.rsplit('.', 1)[0] + '.csv'
            mutants_df.to_csv(csv_output_path, index=False)

            summary = (f"Successfully created mutants-only file!\n\n"
                      f"Input: {self.input_path}\n"
                      f"Output (CSV): {csv_output_path}\n\n"
                      f"Removed {loc_count} Loc## columns\n"
                      f"Kept {len(filtered_columns)} columns\n"
                      f"Total rows: {len(mutants_df)}\n\n"
                      f"NOTE: Saved as CSV to preserve duplicate column names without suffixes.")

            self.progress.emit(100, "Complete!")
            self.finished.emit(summary)

        except Exception as e:
            import traceback
            self.error.emit(f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}")


class ExcelCombinerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.folder_path = None
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("Excel Behavior Data Combiner - Mutants Extractor")
        self.setGeometry(100, 100, 700, 650)
        
        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Title
        title = QLabel("Zebrafish Behavior Data Combiner")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)
        
        # Info label
        info = QLabel("Step 1: Combine Excel files from multiple plates (outputs CSV)\n"
                     "Step 2: Remove Loc## columns from combined file\n"
                     "Step 3: Remove .1, .2, .3 suffixes (preserves nkx2.7 name)")
        info.setAlignment(Qt.AlignCenter)
        info.setStyleSheet("padding: 5px; color: #666;")
        layout.addWidget(info)
        
        # Folder selection
        self.folder_label = QLabel("No folder selected")
        self.folder_label.setStyleSheet("padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
        self.folder_label.setWordWrap(True)
        layout.addWidget(self.folder_label)
        
        self.select_folder_btn = QPushButton("Select Folder with Excel Files")
        self.select_folder_btn.setStyleSheet("padding: 10px; font-size: 14px;")
        self.select_folder_btn.clicked.connect(self.select_folder)
        layout.addWidget(self.select_folder_btn)
        
        # Combine button
        self.combine_btn = QPushButton("Step 1: Combine Excel Files")
        self.combine_btn.setStyleSheet("padding: 10px; font-size: 14px; background-color: #4CAF50; color: white;")
        self.combine_btn.setEnabled(False)
        self.combine_btn.clicked.connect(self.combine_files)
        layout.addWidget(self.combine_btn)

        # Separator
        separator = QLabel("─" * 60)
        separator.setAlignment(Qt.AlignCenter)
        separator.setStyleSheet("color: #ccc; padding: 10px;")
        layout.addWidget(separator)

        # Remove Loc## button
        self.remove_loc_btn = QPushButton("Step 2: Remove Loc## Columns from Combined File")
        self.remove_loc_btn.setStyleSheet("padding: 10px; font-size: 14px; background-color: #2196F3; color: white;")
        self.remove_loc_btn.clicked.connect(self.remove_loc_columns)
        layout.addWidget(self.remove_loc_btn)

        # Separator
        separator2 = QLabel("─" * 60)
        separator2.setAlignment(Qt.AlignCenter)
        separator2.setStyleSheet("color: #ccc; padding: 10px;")
        layout.addWidget(separator2)

        # Remove suffixes button
        self.remove_suffix_btn = QPushButton("Step 3: Remove .1, .2, .3 Suffixes (Keep nkx2.7)")
        self.remove_suffix_btn.setStyleSheet("padding: 10px; font-size: 14px; background-color: #FF9800; color: white;")
        self.remove_suffix_btn.clicked.connect(self.remove_suffixes)
        layout.addWidget(self.remove_suffix_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("padding: 5px;")
        layout.addWidget(self.progress_bar)
        
        # Log area
        log_label = QLabel("Status Log:")
        layout.addWidget(log_label)
        
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet("background-color: #f9f9f9; padding: 5px; font-family: monospace;")
        layout.addWidget(self.log_area)
    
    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder with Excel Files")
        if folder:
            self.folder_path = folder
            self.folder_label.setText(f"Selected: {folder}")
            self.combine_btn.setEnabled(True)
            self.log_area.append(f"Folder selected: {folder}")
    
    def combine_files(self):
        if not self.folder_path:
            return

        # Ask for output file location
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Combined File As",
            "combined_behavior_data.csv",
            "CSV Files (*.csv);;Excel Files (*.xlsx)"
        )

        if not output_path:
            return

        # Disable buttons during processing
        self.select_folder_btn.setEnabled(False)
        self.combine_btn.setEnabled(False)
        self.remove_loc_btn.setEnabled(False)
        self.remove_suffix_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_area.append("\n" + "="*60)
        self.log_area.append("Starting combination process...")
        self.log_area.append("="*60)

        # Start processing thread
        self.thread = ExcelCombinerThread(self.folder_path, output_path)
        self.thread.progress.connect(self.update_progress)
        self.thread.finished.connect(self.on_finished)
        self.thread.error.connect(self.on_error)
        self.thread.start()

    def remove_loc_columns(self):
        # Ask for input file
        input_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Combined File (CSV or Excel)",
            "",
            "Data Files (*.csv *.xlsx *.xls);;CSV Files (*.csv);;Excel Files (*.xlsx *.xls)"
        )

        if not input_path:
            return

        # Ask for output file location
        default_name = Path(input_path).stem + "_mutants_only.csv"
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Mutants-Only File As",
            default_name,
            "CSV Files (*.csv);;Excel Files (*.xlsx)"
        )

        if not output_path:
            return

        # Disable buttons during processing
        self.select_folder_btn.setEnabled(False)
        self.combine_btn.setEnabled(False)
        self.remove_loc_btn.setEnabled(False)
        self.remove_suffix_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_area.append("\n" + "="*60)
        self.log_area.append("Removing Loc## columns...")
        self.log_area.append("="*60)

        # Start processing thread
        self.thread = LocRemoverThread(input_path, output_path)
        self.thread.progress.connect(self.update_progress)
        self.thread.finished.connect(self.on_finished)
        self.thread.error.connect(self.on_error)
        self.thread.start()

    def remove_suffixes(self):
        # Ask for input file
        input_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select File with Suffixes (CSV or Excel)",
            "",
            "Data Files (*.csv *.xlsx *.xls);;CSV Files (*.csv);;Excel Files (*.xlsx *.xls)"
        )

        if not input_path:
            return

        # Ask for output file location
        default_name = Path(input_path).stem + "_no_suffixes.csv"
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Clean File As",
            default_name,
            "CSV Files (*.csv);;Excel Files (*.xlsx)"
        )

        if not output_path:
            return

        # Disable buttons during processing
        self.select_folder_btn.setEnabled(False)
        self.combine_btn.setEnabled(False)
        self.remove_loc_btn.setEnabled(False)
        self.remove_suffix_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_area.append("\n" + "="*60)
        self.log_area.append("Removing suffixes...")
        self.log_area.append("="*60)

        # Start processing thread
        self.thread = SuffixRemoverThread(input_path, output_path)
        self.thread.progress.connect(self.update_progress)
        self.thread.finished.connect(self.on_finished)
        self.thread.error.connect(self.on_error)
        self.thread.start()

    def update_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.log_area.append(message)
        # Auto-scroll to bottom
        self.log_area.verticalScrollBar().setValue(
            self.log_area.verticalScrollBar().maximum()
        )
    
    def on_finished(self, message):
        self.log_area.append("\n" + "="*60)
        self.log_area.append("✓ SUCCESS!")
        self.log_area.append("="*60)
        self.log_area.append(message)
        self.select_folder_btn.setEnabled(True)
        self.combine_btn.setEnabled(True if self.folder_path else False)
        self.remove_loc_btn.setEnabled(True)
        self.remove_suffix_btn.setEnabled(True)

    def on_error(self, message):
        self.log_area.append("\n" + "="*60)
        self.log_area.append("✗ ERROR!")
        self.log_area.append("="*60)
        self.log_area.append(message)
        self.select_folder_btn.setEnabled(True)
        self.combine_btn.setEnabled(True if self.folder_path else False)
        self.remove_loc_btn.setEnabled(True)
        self.remove_suffix_btn.setEnabled(True)
        self.progress_bar.setValue(0)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ExcelCombinerGUI()
    window.show()
    sys.exit(app.exec_())
