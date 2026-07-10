import sys
import os
import shutil
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QComboBox,
    QMessageBox, QProgressBar, QGroupBox, QStatusBar, QTextEdit,
    QDialog, QDialogButtonBox, QScrollArea
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont


# ---------------------------------------------------------------------------
# Build a lookup: stem (no extension, lowercased) -> full filename on disk
# ---------------------------------------------------------------------------
def build_directory_index(image_dir):
    """
    Returns a dict:  lowercase_stem -> actual_filename_with_extension
    e.g. {"img001": "img001.png", "img002": "IMG002.PNG"}
    Handles any extension. If two files share the same stem (different ext),
    the first one found wins (logged as a warning).
    """
    index = {}
    duplicates = []
    for fname in os.listdir(image_dir):
        fpath = os.path.join(image_dir, fname)
        if not os.path.isfile(fpath):
            continue
        stem = os.path.splitext(fname)[0].lower().strip()
        if stem in index:
            duplicates.append(f"  Duplicate stem '{stem}': keeping '{index[stem]}', skipping '{fname}'")
        else:
            index[stem] = fname
    return index, duplicates


def resolve_image(name, index):
    """
    Given a name from the sheet (may or may not have extension),
    find the matching file in the pre-built index.
    Strategy:
      1. Strip extension (if any) from the sheet name, lowercase -> look up in index
      2. If not found, try the full name lowercased as-is (in case it IS the stem)
    Returns the actual filename string, or None if not found.
    """
    name = name.strip()
    stem = os.path.splitext(name)[0].lower().strip()
    if stem in index:
        return index[stem]
    # fallback: maybe the name itself (with extension) is the stem after lowercasing
    if name.lower() in index:
        return index[name.lower()]
    return None


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------
class CopyWorker(QThread):
    progress = pyqtSignal(int)
    # copied_count, missing_list, output_dir, duplicate_warnings
    finished = pyqtSignal(int, list, str, list)
    error = pyqtSignal(str)

    def __init__(self, image_names, image_dir, output_dir):
        super().__init__()
        self.image_names = image_names
        self.image_dir = image_dir
        self.output_dir = output_dir

    def run(self):
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            index, dup_warnings = build_directory_index(self.image_dir)

            total = len(self.image_names)
            copied = 0
            missing = []

            for i, name in enumerate(self.image_names):
                actual_fname = resolve_image(name, index)
                if actual_fname:
                    src = os.path.join(self.image_dir, actual_fname)
                    shutil.copy2(src, self.output_dir)
                    copied += 1
                else:
                    missing.append(name)
                self.progress.emit(int((i + 1) / total * 100))

            self.finished.emit(copied, missing, self.output_dir, dup_warnings)

        except Exception as e:
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Missing-files detail dialog
# ---------------------------------------------------------------------------
class MissingFilesDialog(QDialog):
    def __init__(self, missing, dup_warnings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Missing Files Report")
        self.setMinimumSize(500, 360)
        layout = QVBoxLayout(self)

        info = QLabel(
            f"<b>{len(missing)} file(s)</b> listed in the sheet were not found in "
            "the image directory (after stripping extensions and ignoring case)."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        txt = QTextEdit()
        txt.setReadOnly(True)
        content = "NOT FOUND:\n" + "\n".join(f"  {m}" for m in missing)
        if dup_warnings:
            content += "\n\nDUPLICATE STEM WARNINGS (first file kept):\n" + "\n".join(dup_warnings)
        txt.setPlainText(content)
        layout.addWidget(txt)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class GenotypeExtractor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.df = None
        self._img_col = None
        self._geno_col = None
        self.worker = None
        self.setWindowTitle("Genotype Image Extractor")
        self.setMinimumWidth(640)
        self.setMinimumHeight(460)
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(14)
        root.setContentsMargins(20, 20, 20, 20)

        title = QLabel("Genotype Image Extractor")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        root.addWidget(title)

        # --- Excel file ---
        excel_group = QGroupBox("1. Select Excel / CSV File")
        excel_layout = QHBoxLayout(excel_group)
        self.excel_path = QLineEdit()
        self.excel_path.setPlaceholderText("Path to Excel or CSV file…")
        self.excel_path.setReadOnly(True)
        btn_excel = QPushButton("Browse…")
        btn_excel.setFixedWidth(90)
        btn_excel.clicked.connect(self._browse_excel)
        excel_layout.addWidget(self.excel_path)
        excel_layout.addWidget(btn_excel)
        root.addWidget(excel_group)

        # --- Image directory ---
        img_group = QGroupBox("2. Select Image Directory")
        img_layout = QHBoxLayout(img_group)
        self.img_dir_path = QLineEdit()
        self.img_dir_path.setPlaceholderText("Folder containing the images…")
        self.img_dir_path.setReadOnly(True)
        btn_dir = QPushButton("Browse…")
        btn_dir.setFixedWidth(90)
        btn_dir.clicked.connect(self._browse_dir)
        img_layout.addWidget(self.img_dir_path)
        img_layout.addWidget(btn_dir)
        root.addWidget(img_group)

        # --- Genotype selector ---
        geno_group = QGroupBox("3. Select Genotype")
        geno_layout = QHBoxLayout(geno_group)
        self.geno_combo = QComboBox()
        self.geno_combo.setPlaceholderText("Load an Excel file first…")
        self.geno_combo.setEnabled(False)
        self.count_label = QLabel("")
        self.count_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.geno_combo.currentTextChanged.connect(self._update_count)
        self.geno_combo.currentTextChanged.connect(self._auto_fill_output_name)
        geno_layout.addWidget(self.geno_combo, stretch=1)
        geno_layout.addWidget(self.count_label)
        root.addWidget(geno_group)

        # --- Output subfolder name ---
        out_group = QGroupBox("4. Output Subfolder Name (auto-filled, editable)")
        out_layout = QHBoxLayout(out_group)
        self.out_name = QLineEdit()
        self.out_name.setPlaceholderText("e.g. genotype_WT")
        out_layout.addWidget(self.out_name)
        root.addWidget(out_group)

        # --- Diagnostic: preview match count against selected directory ---
        self.diag_label = QLabel("")
        self.diag_label.setAlignment(Qt.AlignCenter)
        self.diag_label.setStyleSheet("color: #555; font-style: italic;")
        root.addWidget(self.diag_label)

        # --- Extract button + progress ---
        self.extract_btn = QPushButton("Extract Images")
        self.extract_btn.setFixedHeight(38)
        self.extract_btn.setFont(QFont("Arial", 11, QFont.Bold))
        self.extract_btn.setEnabled(False)
        self.extract_btn.clicked.connect(self._run_extraction)
        root.addWidget(self.extract_btn)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        root.addWidget(self.progress)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready.")

    # ------------------------------------------------------------------ helpers

    def _browse_excel(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Excel / CSV File", "",
            "Spreadsheets (*.xlsx *.xlsm *.xls *.csv);;All Files (*)"
        )
        if path:
            self.excel_path.setText(path)
            self._load_excel(path)

    def _browse_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if path:
            self.img_dir_path.setText(path)
            self._refresh_diag()
            self._check_ready()

    def _load_excel(self, path):
        try:
            df = pd.read_csv(path, dtype=str) if path.lower().endswith(".csv") \
                else pd.read_excel(path, dtype=str)

            df.columns = df.columns.str.strip()
            if df.shape[1] < 2:
                raise ValueError("File must have at least two columns (Image Name, Genotype).")

            img_col, geno_col = df.columns[0], df.columns[1]
            df = df[[img_col, geno_col]].copy()
            df[img_col] = df[img_col].astype(str).str.strip()
            df[geno_col] = df[geno_col].astype(str).str.strip()
            df = df[df[img_col].str.lower() != "nan"]
            df = df[df[geno_col].str.lower() != "nan"]

            self.df = df
            self._img_col = img_col
            self._geno_col = geno_col

            genotypes = sorted(df[geno_col].unique().tolist())
            self.geno_combo.clear()
            self.geno_combo.addItems(genotypes)
            self.geno_combo.setEnabled(True)
            self.status_bar.showMessage(
                f"Loaded {len(df)} rows | {len(genotypes)} genotypes | "
                f"Columns: '{img_col}', '{geno_col}'"
            )
            self._refresh_diag()
            self._check_ready()

        except Exception as e:
            QMessageBox.critical(self, "File Error", f"Could not read file:\n{e}")
            self.df = None
            self.geno_combo.setEnabled(False)

    def _update_count(self, genotype):
        if self.df is None or not genotype:
            self.count_label.setText("")
            return
        n = len(self.df[self.df[self._geno_col] == genotype])
        self.count_label.setText(f"({n} image{'s' if n != 1 else ''})")

    def _auto_fill_output_name(self, genotype):
        if genotype:
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in genotype)
            self.out_name.setText(f"genotype_{safe}")
        self._refresh_diag()

    def _refresh_diag(self):
        """Show how many sheet names actually match files in the chosen directory."""
        genotype = self.geno_combo.currentText()
        image_dir = self.img_dir_path.text()
        if not genotype or not image_dir or self.df is None:
            self.diag_label.setText("")
            return
        try:
            index, _ = build_directory_index(image_dir)
            mask = self.df[self._geno_col] == genotype
            names = self.df.loc[mask, self._img_col].tolist()
            matched = sum(1 for n in names if resolve_image(n, index) is not None)
            color = "green" if matched == len(names) else "orange"
            self.diag_label.setText(
                f"<span style='color:{color}'>"
                f"Pre-check: {matched} / {len(names)} sheet names matched to files on disk"
                f"</span>"
            )
        except Exception:
            self.diag_label.setText("")

    def _check_ready(self):
        self.extract_btn.setEnabled(
            self.df is not None and bool(self.img_dir_path.text())
        )

    # ------------------------------------------------------------------ extract

    def _run_extraction(self):
        genotype = self.geno_combo.currentText()
        image_dir = self.img_dir_path.text()
        subfolder = self.out_name.text().strip() or f"genotype_{genotype}"

        if not genotype or not image_dir:
            QMessageBox.warning(self, "Missing Input", "Please select a genotype and image directory.")
            return

        mask = self.df[self._geno_col] == genotype
        image_names = self.df.loc[mask, self._img_col].tolist()

        if not image_names:
            QMessageBox.information(self, "No Images", f"No images found for genotype: {genotype}")
            return

        output_dir = os.path.join(image_dir, subfolder)

        if os.path.exists(output_dir):
            reply = QMessageBox.question(
                self, "Folder Exists",
                f"Output folder already exists:\n{output_dir}\n\nOverwrite / merge?",
                QMessageBox.Yes | QMessageBox.Cancel
            )
            if reply != QMessageBox.Yes:
                return

        self.extract_btn.setEnabled(False)
        self.progress.setValue(0)
        self.status_bar.showMessage(f"Copying {len(image_names)} images…")

        self.worker = CopyWorker(image_names, image_dir, output_dir)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self._on_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_done(self, copied, missing, output_dir, dup_warnings):
        self.extract_btn.setEnabled(True)
        self._refresh_diag()

        if missing or dup_warnings:
            summary = (
                f"<b>{copied}</b> image(s) copied to:<br><i>{output_dir}</i><br><br>"
                f"<b style='color:orange'>{len(missing)}</b> file(s) were not found "
                f"(see details below)."
            )
            QMessageBox.information(self, "Extraction Complete", summary)
            dlg = MissingFilesDialog(missing, dup_warnings, self)
            dlg.exec_()
        else:
            QMessageBox.information(
                self, "Extraction Complete",
                f"<b>{copied}</b> image(s) copied to:<br><i>{output_dir}</i>"
            )

        self.status_bar.showMessage(
            f"Done. Copied: {copied}  |  Not found: {len(missing)}"
        )

    def _on_error(self, msg):
        self.extract_btn.setEnabled(True)
        self.status_bar.showMessage("Error during extraction.")
        QMessageBox.critical(self, "Error", f"An error occurred:\n{msg}")


# ---------------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = GenotypeExtractor()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()