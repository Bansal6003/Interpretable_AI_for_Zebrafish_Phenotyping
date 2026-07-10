from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QTabWidget,
                           QLabel, QApplication)
import sys
import os
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from DanioV16_core import DanioAnalyzerGUI
from LLM_based_summariztion import ClaudeAnalysis



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Danio Analyzer Platform")
        self.setGeometry(100, 100, 1200, 800)
        self.setup_ui()
        
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = QLabel("Danio Analyzer Platform")
        title.setFont(QFont('Arial', 24, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Create tabs
        tabs = QTabWidget()
        
        # Classification and Segmentation tab
        class_seg_tab = DanioAnalyzerGUI()
        tabs.addTab(class_seg_tab, "Classification & Segmentation")
      
        
        # # NLP & LLM Analysis tab
        nlp_tab = ClaudeAnalysis()
        tabs.addTab(nlp_tab, "NLP & LLM Analysis")
        
        # # Gene Comparison tab (new)
        # self.gene_comparison_tab = GeneComparisonTab()
        # tabs.addTab(self.gene_comparison_tab, "Gene Comparison")
        
        layout.addWidget(tabs)
        
    def launch_advanced_plot_window(self):
        """Launch an advanced plot window from the main UI"""
        from advanced_plotting import AdvancedGenePlotter
        
        # Check if the gene comparison tab has data
        if not hasattr(self.gene_comparison_tab, 'gene_data') or not self.gene_comparison_tab.gene_data:
            return
            
        # Get the currently selected gene
        gene = self.gene_comparison_tab.gene_combo.currentText()
        if not gene:
            return
        
        # Create plotter and show available plots
        plotter = AdvancedGenePlotter(self.gene_comparison_tab)
        
        # Show time heatmap
        plotter.plot_time_series_heatmap(
            self.gene_comparison_tab.gene_data, 
            self.gene_comparison_tab.wt_data, 
            gene
        )
        
        # Show response curves
        plotter.plot_aligned_time_series(
            self.gene_comparison_tab.gene_data, 
            self.gene_comparison_tab.wt_data, 
            gene
        )
        
        # Show volcano plot
        plotter.plot_volcano(
            self.gene_comparison_tab.summary_data,
            gene
        )

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()