import sys
import os
import re
import glob
import threading

import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTextEdit, QProgressBar,
    QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon, QTextCursor


# ── Worker thread ────────────────────────────────────────────────────────────

class MergeWorker(QThread):
    log       = pyqtSignal(str, str)   # message, level (info/ok/warn/error)
    progress  = pyqtSignal(int)
    finished  = pyqtSignal(bool, str)  # success, output_path

    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir

    def run(self):
        try:
            root_dir = os.path.abspath(self.root_dir)
            folders  = self._find_numbered_folders(root_dir)

            if not folders:
                self.log.emit("No folders ending in numbers 1–15 were found.", "warn")
                self.finished.emit(False, "")
                return

            self.log.emit(f"Found {len(folders)} eligible folder(s).", "info")
            all_dfs    = []
            total      = len(folders)

            for idx, (num, folder) in enumerate(folders, 1):
                pattern = os.path.join(folder, "*_detailed*.xlsx")
                files   = sorted(glob.glob(pattern))

                if not files:
                    self.log.emit(f"[{num:>2}] {os.path.basename(folder)}  — no _detailed file found, skipping.", "warn")
                else:
                    for filepath in files:
                        rel = os.path.relpath(filepath, root_dir)
                        self.log.emit(f"[{num:>2}] Reading  {rel}", "info")
                        try:
                            sheets = pd.read_excel(filepath, sheet_name=None)
                            for sheet_name, df in sheets.items():
                                df.insert(0, "_source_folder", os.path.basename(folder))
                                df.insert(1, "_source_file",   os.path.basename(filepath))
                                df.insert(2, "_source_sheet",  sheet_name)
                                all_dfs.append(df)
                        except Exception as e:
                            self.log.emit(f"      ERROR reading {rel}: {e}", "error")

                self.progress.emit(int(idx / total * 90))

            if not all_dfs:
                self.log.emit("No data collected — nothing to save.", "warn")
                self.finished.emit(False, "")
                return

            merged      = pd.concat(all_dfs, ignore_index=True)
            output_path = os.path.join(root_dir, "merged_detailed.xlsx")
            merged.to_excel(output_path, index=False)
            self.progress.emit(100)
            self.log.emit(
                f"Done!  {len(all_dfs)} sheet(s) merged  →  {output_path}  "
                f"({len(merged):,} rows, {len(merged.columns)} columns)",
                "ok"
            )
            self.finished.emit(True, output_path)

        except Exception as e:
            self.log.emit(f"Unexpected error: {e}", "error")
            self.finished.emit(False, "")

    def _find_numbered_folders(self, root_dir):
        result = []
        for entry in os.scandir(root_dir):
            if entry.is_dir():
                m = re.search(r'(\d+)$', entry.name)
                if m:
                    n = int(m.group(1))
                    if 1 <= n <= 15:
                        result.append((n, entry.path))
        return sorted(result, key=lambda x: x[0])


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.root_dir = ""
        self.worker   = None
        self._build_ui()
        self._apply_stylesheet()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        self.setWindowTitle("Excel Merger — _detailed files")
        self.setMinimumSize(760, 560)

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(28, 24, 28, 20)
        root_layout.setSpacing(16)

        # ── Header ────────────────────────────────────────────────────────────
        header = QLabel("Excel Merger")
        header.setObjectName("header")
        sub = QLabel("Collects every  <i>*_detailed*.xlsx</i>  from folders numbered 1 – 15 and merges them into one file.")
        sub.setObjectName("sub")
        sub.setWordWrap(True)
        root_layout.addWidget(header)
        root_layout.addWidget(sub)

        # ── Divider ───────────────────────────────────────────────────────────
        root_layout.addWidget(self._divider())

        # ── Folder picker row ────────────────────────────────────────────────
        picker_row = QHBoxLayout()
        picker_row.setSpacing(10)

        self.folder_label = QLabel("No folder selected")
        self.folder_label.setObjectName("folderLabel")
        self.folder_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        browse_btn = QPushButton("Browse…")
        browse_btn.setObjectName("browseBtn")
        browse_btn.setFixedWidth(110)
        browse_btn.clicked.connect(self._browse)

        picker_row.addWidget(self.folder_label)
        picker_row.addWidget(browse_btn)
        root_layout.addLayout(picker_row)

        # ── Progress bar ──────────────────────────────────────────────────────
        self.progress = QProgressBar()
        self.progress.setObjectName("progressBar")
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(8)
        root_layout.addWidget(self.progress)

        # ── Log area ──────────────────────────────────────────────────────────
        self.log_box = QTextEdit()
        self.log_box.setObjectName("logBox")
        self.log_box.setReadOnly(True)
        self.log_box.setFont(QFont("Courier New", 10))
        root_layout.addWidget(self.log_box, 1)

        # ── Bottom row ────────────────────────────────────────────────────────
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(10)

        self.status_label = QLabel("")
        self.status_label.setObjectName("statusLabel")

        self.run_btn = QPushButton("▶  Run Merge")
        self.run_btn.setObjectName("runBtn")
        self.run_btn.setFixedWidth(150)
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._run)

        self.clear_btn = QPushButton("Clear log")
        self.clear_btn.setObjectName("clearBtn")
        self.clear_btn.setFixedWidth(100)
        self.clear_btn.clicked.connect(self._clear_log)

        bottom_row.addWidget(self.status_label, 1)
        bottom_row.addWidget(self.clear_btn)
        bottom_row.addWidget(self.run_btn)
        root_layout.addLayout(bottom_row)

    def _divider(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setObjectName("divider")
        return line

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Root Directory", os.path.expanduser("~"))
        if folder:
            self.root_dir = folder
            display = folder if len(folder) < 70 else "…" + folder[-67:]
            self.folder_label.setText(display)
            self.folder_label.setToolTip(folder)
            self.run_btn.setEnabled(True)
            self._set_status("")

    def _run(self):
        if not self.root_dir:
            return
        self.log_box.clear()
        self.progress.setValue(0)
        self.run_btn.setEnabled(False)
        self._set_status("Running…")

        self.worker = MergeWorker(self.root_dir)
        self.worker.log.connect(self._append_log)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _on_finished(self, success, output_path):
        self.run_btn.setEnabled(True)
        if success:
            self._set_status(f"✓  Saved: {os.path.basename(output_path)}", ok=True)
        else:
            self._set_status("⚠  Merge failed — see log for details.", ok=False)

    def _append_log(self, message, level):
        colors = {
            "info":  "#c9d1d9",
            "ok":    "#3fb950",
            "warn":  "#d29922",
            "error": "#f85149",
        }
        color = colors.get(level, "#c9d1d9")
        self.log_box.append(f'<span style="color:{color};">{message}</span>')
        self.log_box.moveCursor(QTextCursor.End)

    def _clear_log(self):
        self.log_box.clear()
        self.progress.setValue(0)
        self._set_status("")

    def _set_status(self, text, ok=None):
        self.status_label.setText(text)
        if ok is True:
            self.status_label.setStyleSheet("color: #3fb950; font-weight: 600;")
        elif ok is False:
            self.status_label.setStyleSheet("color: #f85149; font-weight: 600;")
        else:
            self.status_label.setStyleSheet("color: #8b949e;")

    # ── Stylesheet ─────────────────────────────────────────────────────────────

    def _apply_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #0d1117;
                color: #c9d1d9;
            }

            QLabel#header {
                font-size: 22px;
                font-weight: 700;
                color: #e6edf3;
                letter-spacing: 1px;
            }

            QLabel#sub {
                font-size: 12px;
                color: #8b949e;
                margin-bottom: 4px;
            }

            QFrame#divider {
                color: #21262d;
            }

            QLabel#folderLabel {
                background: #161b22;
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 7px 12px;
                color: #8b949e;
                font-size: 12px;
            }

            QPushButton#browseBtn {
                background: #21262d;
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 7px 14px;
                color: #c9d1d9;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton#browseBtn:hover {
                background: #30363d;
                border-color: #8b949e;
            }
            QPushButton#browseBtn:pressed {
                background: #161b22;
            }

            QPushButton#runBtn {
                background: #238636;
                border: 1px solid #2ea043;
                border-radius: 6px;
                padding: 8px 18px;
                color: #ffffff;
                font-size: 13px;
                font-weight: 700;
            }
            QPushButton#runBtn:hover {
                background: #2ea043;
            }
            QPushButton#runBtn:pressed {
                background: #1a7f37;
            }
            QPushButton#runBtn:disabled {
                background: #21262d;
                border-color: #30363d;
                color: #484f58;
            }

            QPushButton#clearBtn {
                background: transparent;
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 7px 14px;
                color: #8b949e;
                font-size: 12px;
            }
            QPushButton#clearBtn:hover {
                border-color: #8b949e;
                color: #c9d1d9;
            }

            QProgressBar#progressBar {
                background: #21262d;
                border: none;
                border-radius: 4px;
            }
            QProgressBar#progressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #388bfd, stop:1 #3fb950);
                border-radius: 4px;
            }

            QTextEdit#logBox {
                background: #161b22;
                border: 1px solid #21262d;
                border-radius: 8px;
                padding: 10px;
                color: #c9d1d9;
                font-size: 12px;
                selection-background-color: #264f78;
            }

            QLabel#statusLabel {
                font-size: 12px;
                color: #8b949e;
            }

            QScrollBar:vertical {
                background: #161b22;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #30363d;
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #484f58;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
        """)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())