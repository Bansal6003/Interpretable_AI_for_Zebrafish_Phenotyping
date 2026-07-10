
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import os
import json
import re
from datetime import datetime
import sys
import traceback

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, 
                             QPushButton, QLabel, QFileDialog, QApplication,
                             QProgressBar, QComboBox, QTabWidget, QSplitter,
                             QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
                             QLineEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

# Langchain imports
# from langchain.llms import Anthropic
# from langchain.chat_models import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage
# from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
# from langchain.chains import LLMChain
# from langchain.output_parsers import PydanticOutputParser
# from langchain.pydantic_v1 import BaseModel, Field, validator
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.document_loaders import DataFrameLoader
# from langchain.chains import RetrievalQA
# from langchain.callbacks.manager import CallbackManager
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from langchain.memory import ConversationBufferMemory
# from typing import List, Dict, Any, Optional


class ProgressCallback:
    """Callback handler for reporting progress during LLM operations"""
    def __init__(self, progress_signal=None):
        self.progress_signal = progress_signal
        self.token_count = 0
        self.last_progress = 0
    
    def on_llm_start(self, *args, **kwargs):
        if self.progress_signal:
            self.progress_signal.emit(5)
    
    def on_llm_new_token(self, token, *args, **kwargs):
        self.token_count += 1
        # Update progress every 10 tokens, approximating % completion
        if self.token_count % 10 == 0 and self.progress_signal:
            progress = min(90, 5 + self.token_count // 10)
            if progress > self.last_progress:
                self.progress_signal.emit(progress)
                self.last_progress = progress
    
    def on_llm_end(self, *args, **kwargs):
        if self.progress_signal:
            self.progress_signal.emit(100)


class DataAnalysisThread(QThread):
    """Thread for running Claude-based analysis operations without freezing the GUI"""
    update_progress = pyqtSignal(int)
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, operation, data, api_key=None):
        super().__init__()
        self.operation = operation
        self.data = data
        self.api_key = api_key
        self.callback = ProgressCallback(self.update_progress)
        
    def run(self):
        try:
            result = {}
            
            # Initialize Claude client
            if not self.api_key:
                # Look for API key in environment variable
                self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
                
                if not self.api_key:
                    self.error_occurred.emit("Claude API key not found. Please set ANTHROPIC_API_KEY environment variable or provide it in the settings.")
                    return
            
            # Initialize Anthropic client
            import anthropic
            self.anthropic_client = anthropic.Anthropic(api_key=self.api_key)
            
            if self.operation == "data_summary":
                result = self.run_data_summary()
            elif self.operation == "experiment_analysis":
                result = self.run_experiment_analysis()
            elif self.operation == "gene_analysis":
                result = self.run_gene_analysis()
            elif self.operation == "custom_query":
                result = self.run_custom_query()
            elif self.operation == "phenotype_prediction":
                result = self.run_phenotype_prediction()
            elif self.operation == "batch_analysis":
                # Inline implementation of batch analysis with user-defined queries
                print("Starting batch analysis...")
                
                if not isinstance(self.data, dict) or 'queries' not in self.data or 'dataframe' not in self.data:
                    result = {"error": "Invalid data format for batch analysis. Expected dict with 'queries' and 'dataframe'."}
                else:
                    queries = self.data['queries']
                    df = self.data['dataframe']
                    data_context = self.dataframe_to_context(df)
                    
                    self.update_progress.emit(10)
                    
                    # Process each user-defined query
                    results_dict = {}
                    
                    total_queries = len(queries)
                    for i, query in enumerate(queries):
                        # Create a unique ID for this query
                        query_id = f"query-{i+1}"
                        
                        # Create a prompt with data context
                        prompt = f"Here is a dataset to analyze:\n\n{data_context}\n\n{query}"
                        
                        # Update progress proportionally
                        progress = 10 + int(80 * (i / total_queries))
                        self.update_progress.emit(progress)
                        
                        # Process query
                        message = self.anthropic_client.messages.create(
                            model="claude-3-haiku-20240307",
                            max_tokens=800,
                            temperature=0.1,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        results_dict[query_id] = {
                            "query": query,
                            "result": message.content[0].text
                        }
                    
                    # Combine the results into a single markdown document
                    combined_analysis = "# Multiple Insights Analysis\n\n"
                    
                    for query_id, query_result in results_dict.items():
                        combined_analysis += f"## {query_result['query']}\n\n"
                        combined_analysis += f"{query_result['result']}\n\n"
                        combined_analysis += "---\n\n"
                    
                    result = {
                        "type": "batch_analysis",
                        "result": combined_analysis,
                        "individual_results": results_dict,
                        "metadata": {
                            "num_records": len(df),
                            "num_queries": len(queries),
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                    }
            else:
                result = {"error": f"Unknown operation: {self.operation}"}
                
            self.result_ready.emit(result)
        except Exception as e:
            error_msg = f"Error: {str(e)}\n\n"
            error_msg += traceback.format_exc()
            self.error_occurred.emit(error_msg)
    
    def dataframe_to_context(self, df):
        """Convert DataFrame to a readable text context for the LLM including all rows and columns"""
        # Basic info
        context = f"Dataset contains {len(df)} rows and {len(df.columns)} columns.\n\n"
        context += f"Columns: {', '.join(df.columns.tolist())}\n\n"
        
        # All data (all rows)
        context += "Full dataset:\n"
        context += df.to_string() + "\n\n"
        
        # Statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            context += "Statistics for numeric columns:\n"
            context += df[numeric_cols].describe().to_string() + "\n\n"
        
        # Unique values for all categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            context += "Categories (all unique values for each column):\n"
            for col in categorical_cols:  # Include all categorical columns
                unique_vals = df[col].unique()  # Include all unique values
                context += f"{col}: {', '.join(map(str, unique_vals))}\n"
        
        return context
    
    def run_data_summary(self):
        """Generate a comprehensive summary of the uploaded data"""
        self.update_progress.emit(10)
        
        if not isinstance(self.data, pd.DataFrame):
            return {"error": "Invalid data format. Expected DataFrame."}
        
        df = self.data
        
        # Get data context
        data_context = self.dataframe_to_context(df)
        
        # Prepare prompt
        system_prompt = """You are a scientific data analyst specializing in biological datasets. 
        Provide a clear, comprehensive summary of the dataset described below.
        Focus on key patterns, notable features, potential research questions, and data quality issues.
        Format your response with markdown headers and bullet points for readability."""
        
        human_prompt = f"""Here is a dataset to summarize:
        
        {data_context}
        
        Please provide:
        1. An overview of what this dataset contains
        2. Key observations about the data structure and content
        3. Potential research questions this data could answer
        4. Any data quality concerns or limitations
        5. Recommendations for further analysis"""
        
        self.update_progress.emit(30)
        
        # Generate summary using Anthropic's API
        message = self.anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            temperature=0.1,
            system=system_prompt,
            messages=[
                {"role": "user", "content": human_prompt}
            ]
        )
        
        # Extract the content
        summary = message.content[0].text
        
        self.update_progress.emit(100)
        
        return {
            "type": "data_summary",
            "result": summary,
            "metadata": {
                "num_records": len(df),
                "num_columns": len(df.columns),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
    
    def run_data_summary(self):
        """Generate a comprehensive summary of the uploaded data"""
        self.update_progress.emit(10)
        
        if not isinstance(self.data, pd.DataFrame):
            return {"error": "Invalid data format. Expected DataFrame."}
        
        df = self.data
        
        # Get data context
        data_context = self.dataframe_to_context(df)
        
        # Prepare prompt
        system_prompt = """You are a scientific data analyst specializing in biological datasets. 
        Provide a clear, comprehensive summary of the dataset described below.
        Focus on key patterns, notable features, potential research questions, and data quality issues.
        Format your response with markdown headers and bullet points for readability."""
        
        human_prompt = f"""Here is a dataset to summarize:
        
        {data_context}
        
        Please provide:
        1. An overview of what this dataset contains
        2. Key observations about the data structure and content
        3. Potential research questions this data could answer
        4. Any data quality concerns or limitations
        5. Recommendations for further analysis"""
        
        self.update_progress.emit(30)
        
        # Generate summary using Anthropic's API
        message = self.anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            temperature=0.1,
            system=system_prompt,
            messages=[
                {"role": "user", "content": human_prompt}
            ]
        )
        
        # Extract the content
        summary = message.content[0].text
        
        self.update_progress.emit(100)
        
        return {
            "type": "data_summary",
            "result": summary,
            "metadata": {
                "num_records": len(df),
                "num_columns": len(df.columns),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        def run_experiment_analysis(self):
            """Analyze experimental data to extract insights and patterns"""
            self.update_progress.emit(10)
            
            if not isinstance(self.data, pd.DataFrame):
                return {"error": "Invalid data format. Expected DataFrame."}
            
            df = self.data
            
            # Get data context
            data_context = self.dataframe_to_context(df)
            
            # Prepare prompt
            system_prompt = """You are a scientific data analyst specializing in experimental biology data.
            Analyze the experimental data described below and provide insights, relationships between variables,
            and potential conclusions that can be drawn from the experiments.
            Focus on identifying patterns, experimental outcomes, and scientific implications.
            Format your response with markdown sections for clarity."""
            
            human_prompt = f"""Here is an experimental dataset to analyze:
            
            {data_context}
            
            Please provide:
            1. A detailed analysis of experimental outcomes
            2. Statistical patterns and relationships between variables
            3. Potential scientific conclusions based on the data
            4. Suggestions for follow-up experiments
            5. Limitations of the current experimental design"""
            
            # Create messages
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            self.update_progress.emit(30)
            
            # Generate analysis
            response = self.chat_model.invoke(messages)
            analysis = response.content
            
            self.update_progress.emit(100)
            
            return {
                "type": "experiment_analysis",
                "result": analysis,
                "metadata": {
                    "num_records": len(df),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }
    
    def run_gene_analysis(self):
        """Analyze gene data to find patterns and relationships"""
        self.update_progress.emit(10)
        
        if not isinstance(self.data, pd.DataFrame):
            return {"error": "Invalid data format. Expected DataFrame."}
        
        df = self.data
        
        # Check if we have gene-related columns
        gene_related_cols = [col for col in df.columns 
                           if any(term in col.lower() for term in 
                                 ['gene', 'allele', 'mutation', 'expression', 'phenotype'])]
        
        if not gene_related_cols:
            return {"error": "No gene-related columns found in the dataset."}
        
        # Get data context with focus on gene data
        gene_df = df[gene_related_cols] if len(gene_related_cols) > 0 else df
        data_context = self.dataframe_to_context(gene_df)
        
        # Prepare prompt
        system_prompt = """You are a genetics expert specializing in gene function analysis.
        Analyze the gene data provided below to identify patterns, functional relationships,
        and potential gene interactions or pathways.
        Focus on identifying gene clusters, expression patterns, and phenotypic associations.
        Format your response using markdown with clear sections and bullet points."""
        
        human_prompt = f"""Here is gene-related data to analyze:
        
        {data_context}
        
        Please provide:
        1. An analysis of gene patterns and clusters in the data
        2. Functional relationships between genes if present
        3. Phenotypic associations and their significance
        4. Potential pathways or processes implicated by the gene data
        5. Recommendations for further genetic analysis"""
        
        # Create messages
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        self.update_progress.emit(30)
        
        # Generate analysis
        response = self.chat_model.invoke(messages)
        analysis = response.content
        
        self.update_progress.emit(100)
        
        # Extract gene clusters if possible using a structured approach
        gene_clusters = self.extract_gene_clusters_from_analysis(analysis, df)
        
        return {
            "type": "gene_analysis",
            "result": analysis,
            "clusters": gene_clusters,
            "metadata": {
                "num_records": len(df),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
    
    def extract_gene_clusters_from_analysis(self, analysis_text, df):
        """Try to extract gene clusters from the analysis text"""
        # This is a simplistic approach - in a real app, we would use a more structured approach
        # with the output parser from langchain
        clusters = {}
        
        # Try to find section about clusters
        cluster_sections = re.findall(r'(cluster|group|category)\s*\d+[:\s]+(.*?)(?=\n\s*(?:cluster|group|category)\s*\d+[:\s]+|\Z)', 
                                     analysis_text, re.IGNORECASE | re.DOTALL)
        
        if cluster_sections:
            for i, (_, section_text) in enumerate(cluster_sections):
                # Try to extract gene names from text
                potential_genes = []
                
                # If we have a Gene column, use those values for matching
                if 'Gene' in df.columns:
                    genes_in_df = df['Gene'].astype(str).unique()
                    for gene in genes_in_df:
                        if gene and len(gene) > 1 and gene in section_text:
                            potential_genes.append(gene)
                
                # If we didn't find genes but have gene-like patterns, extract those
                if not potential_genes:
                    # Look for gene naming patterns (simplified)
                    gene_pattern = r'\b[A-Za-z]+\d*[A-Za-z]*\b'
                    found_genes = re.findall(gene_pattern, section_text)
                    # Filter out common words
                    common_words = {'the', 'and', 'this', 'that', 'with', 'from', 'have', 'genes'}
                    potential_genes = [g for g in found_genes if g.lower() not in common_words and len(g) > 2]
                
                clusters[f"Cluster {i+1}"] = {
                    "genes": potential_genes[:10],  # Limit to 10 genes per cluster
                    "description": section_text.strip()[:100] + "..." if len(section_text) > 100 else section_text.strip()
                }
        
        return clusters
    
    def run_phenotype_prediction(self):
        """Predict phenotypes based on gene and experimental data"""
        self.update_progress.emit(10)
        
        if not isinstance(self.data, pd.DataFrame):
            return {"error": "Invalid data format. Expected DataFrame."}
        
        df = self.data
        
        # Get data context
        data_context = self.dataframe_to_context(df)
        
        # Prepare prompt
        system_prompt = """You are a genetic phenotype prediction expert.
        Analyze the provided data and predict potential phenotypes based on genetic markers,
        gene expressions, or experimental conditions described in the data.
        Provide detailed reasoning for your predictions and confidence levels where appropriate.
        Format your response with markdown for readability."""
        
        human_prompt = f"""Here is gene and experimental data for phenotype prediction:
        
        {data_context}
        
        Please provide:
        1. Predicted phenotypes based on the data
        2. Evidence supporting each prediction
        3. Confidence level for each prediction
        4. Potential alternative phenotypes
        5. Recommendations for experimental validation"""
        
        # Create messages
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        self.update_progress.emit(30)
        
        # Generate predictions
        response = self.chat_model.invoke(messages)
        predictions = response.content
        
        self.update_progress.emit(100)
        
        return {
            "type": "phenotype_prediction",
            "result": predictions,
            "metadata": {
                "num_records": len(df),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
    
    def run_custom_query(self):
        """Run a custom query against the data using Claude"""
        self.update_progress.emit(10)
        
        if not isinstance(self.data, dict) or 'query' not in self.data or 'dataframe' not in self.data:
            return {"error": "Invalid data format for custom query. Expected dict with 'query' and 'dataframe'."}
        
        query = self.data['query']
        df = self.data['dataframe']
        
        # Get data context
        data_context = self.dataframe_to_context(df)
        
        # Prepare prompt
        system_prompt = """You are a scientific data analyst with expertise in analyzing biological and experimental data.
        You will be given a dataset description and a specific question about the data.
        Provide a thorough, evidence-based answer to the question using only the information in the provided data.
        Be clear about what the data can and cannot tell us, and avoid making claims without evidence.
        Format your response with markdown for readability."""
        
        human_prompt = f"""Here is a dataset:
        
        {data_context}
        
        My question is: {query}
        
        Please provide a detailed answer based on this data."""
        
        # Create messages
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        self.update_progress.emit(30)
        
        # Generate response
        response = self.chat_model.invoke(messages)
        answer = response.content
        
        self.update_progress.emit(100)
        
        return {
            "type": "custom_query",
            "query": query,
            "result": answer,
            "metadata": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }


class ClaudeAnalysis(QWidget):
    def __init__(self):
        super().__init__()
        # Initialize the api_key attribute before setup_ui
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.current_data = None
        self.setup_ui()
       
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Claude Powered Data Analysis")
        title.setFont(QFont('Arial', 16, QFont.Bold))
        layout.addWidget(title)
        
        # Create a splitter for resizable panels
        splitter = QSplitter(Qt.Vertical)
        
        # Top panel for data loading and operation selection
        top_panel = QWidget()
        top_layout = QVBoxLayout(top_panel)
        
        # API Key section
        api_section = QWidget()
        api_layout = QHBoxLayout(api_section)
        
        api_layout.addWidget(QLabel("Claude API Key:"))
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setPlaceholderText("Enter API key or set ANTHROPIC_API_KEY env variable")
        if self.api_key:
            self.api_key_input.setText("*" * 10 + " (from environment)")
        api_layout.addWidget(self.api_key_input)
        
        api_button = QPushButton("Save Key")
        api_button.clicked.connect(self.save_api_key)
        api_layout.addWidget(api_button)
        
        top_layout.addWidget(api_section)
        
        # Data loading section
        data_section = QWidget()
        data_layout = QHBoxLayout(data_section)
        
        self.load_data_btn = QPushButton("Load Data File")
        self.load_data_btn.clicked.connect(self.load_data)
        
        self.data_info_label = QLabel("No data loaded")
        
        data_layout.addWidget(self.load_data_btn)
        data_layout.addWidget(self.data_info_label)
        top_layout.addWidget(data_section)
        
        # Operation selection
        operation_layout = QHBoxLayout()
        
        self.operation_combo = QComboBox()
        self.operation_combo.addItems([
            "Data Summary",
            "Experiment Analysis",
            "Gene Analysis",
            "Phenotype Prediction",
            "Custom Query",
            "Batch Analysis (Multiple Insights)"  # Add this new option
        ])
        
        self.run_button = QPushButton("Run Analysis")
        self.run_button.clicked.connect(self.run_selected_operation)
        
        operation_layout.addWidget(QLabel("Operation:"))
        operation_layout.addWidget(self.operation_combo)
        operation_layout.addWidget(self.run_button)
        
        top_layout.addLayout(operation_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        top_layout.addWidget(self.progress_bar)
        
        # Add query input for custom queries
        self.query_section = QWidget()
        query_layout = QVBoxLayout(self.query_section)
        query_layout.addWidget(QLabel("Enter your research question:"))
        
        self.query_input = QTextEdit()
        self.query_input.setPlaceholderText("e.g., What patterns do you see in the gene expression data? or Compare the mutant vs. wild type phenotypes.")
        self.query_input.setMaximumHeight(80)
        query_layout.addWidget(self.query_input)
        
        self.query_section.setVisible(False)
        top_layout.addWidget(self.query_section)
        
        # Bottom panel with tabs for different visualizations
        bottom_panel = QTabWidget()
        
        # Text output tab
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        bottom_panel.addTab(self.text_output, "Analysis Results")
        
        # Visualization tab
        visualization_widget = QWidget()
        viz_layout = QVBoxLayout(visualization_widget)
        
        self.figure = plt.figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        viz_layout.addWidget(self.canvas)
        
        bottom_panel.addTab(visualization_widget, "Visualization")
        
        # Table results tab
        self.results_table = QTableWidget()
        bottom_panel.addTab(self.results_table, "Table View")
        
        # Add panels to splitter
        splitter.addWidget(top_panel)
        splitter.addWidget(bottom_panel)
        splitter.setSizes([200, 400])  # Set initial sizes
        
        layout.addWidget(splitter)
        
        
        self.batch_section = QWidget()
        batch_layout = QVBoxLayout(self.batch_section)
        batch_layout.addWidget(QLabel("Define custom insights to extract from your data (one per line):"))
           
        self.batch_queries_input = QTextEdit()
        self.batch_queries_input.setPlaceholderText("Enter one query per line, for example:\n"
                                                  "Summarize the key findings in this dataset\n"
                                                  "Identify patterns and relationships between variables\n"
                                                  "What recommendations would you make based on this data?")
        self.batch_queries_input.setMinimumHeight(100)
        batch_layout.addWidget(self.batch_queries_input)
        
        # Set default queries
        default_queries = "Summarize the key findings in this dataset\n" \
                          "Identify patterns and relationships between variables\n" \
                          "What recommendations would you make based on this data?"
        self.batch_queries_input.setText(default_queries)
        
        self.batch_section.setVisible(False)
        top_layout.addWidget(self.batch_section)
   
        # Connect signals
        self.operation_combo.currentTextChanged.connect(self.on_operation_changed)
        
        
    def save_api_key(self):
        """Save the Claude API key"""
        key_text = self.api_key_input.text()
        if key_text and not key_text.startswith("*"):
            self.api_key = key_text
            self.api_key_input.setText("*" * 10 + " (saved)")
            QMessageBox.information(self, "API Key", "API key saved for this session.")
        else:
            QMessageBox.warning(self, "API Key", "Please enter a valid API key.")
    
    def on_operation_changed(self, operation):
        """Show/hide appropriate input sections based on operation"""
        # Hide all sections first
        self.query_section.setVisible(False)
        self.batch_section.setVisible(False)
        
        # Show appropriate section based on operation
        if operation == "Custom Query":
            self.query_section.setVisible(True)
        elif operation == "Batch Analysis (Multiple Insights)":
            self.batch_section.setVisible(True)
    
    def load_data(self):
        """Load data from Excel or CSV file"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Data File", "", "Excel files (*.xlsx);;CSV files (*.csv)")
            
            if not file_path:
                return
                
            if file_path.endswith('.xlsx'):
                self.current_data = pd.read_excel(file_path)
            elif file_path.endswith('.csv'):
                self.current_data = pd.read_csv(file_path)
            else:
                raise ValueError("Unsupported file format")
                
            # Update info label
            self.data_info_label.setText(f"Loaded: {os.path.basename(file_path)} - {len(self.current_data)} records")
            
            # Show brief data summary in text output
            self.text_output.clear()
            self.text_output.append(f"# Data Summary for {os.path.basename(file_path)}\n")
            self.text_output.append(f"Records: {len(self.current_data)}\n")
            self.text_output.append(f"Columns: {', '.join(self.current_data.columns)}\n\n")
            
            if len(self.current_data) > 0:
                # Show sample data
                self.text_output.append("## Sample Data (First 5 rows)\n")
                self.text_output.append(f"```\n{self.current_data.head(5).to_string()}\n```\n")
                
                # Show column statistics
                self.text_output.append("## Column Statistics\n")
                numeric_cols = self.current_data.select_dtypes(include=[np.number]).columns
                
                for col in numeric_cols:
                    self.text_output.append(f"**{col}**:")
                    self.text_output.append(f"  - Min: {self.current_data[col].min()}")
                    self.text_output.append(f"  - Max: {self.current_data[col].max()}")
                    self.text_output.append(f"  - Mean: {self.current_data[col].mean():.2f}")
                    self.text_output.append(f"  - Std: {self.current_data[col].std():.2f}\n")
                
        except Exception as e:
            self.data_info_label.setText(f"Error loading data: {str(e)}")
            self.text_output.clear()
            self.text_output.append(f"Error loading data: {str(e)}")
    
    def run_selected_operation(self):
        """Run the selected operation with Claude"""
        if self.current_data is None:
            QMessageBox.warning(self, "Warning", "Please load data first.")
            return
            
        operation_text = self.operation_combo.currentText()
        
        # Convert to function name format
        if operation_text == "Batch Analysis (Multiple Insights)":
            operation = "batch_analysis"
        else:
            operation = operation_text.lower().replace(" ", "_")
        
        # Clear previous outputs
        self.text_output.clear()
        self.figure.clear()
        self.canvas.draw()
        self.results_table.clear()
        
        # Show progress bar
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        
        # Prepare data based on operation
        if operation == "custom_query":
            query = self.query_input.toPlainText().strip()
            if not query:
                QMessageBox.warning(self, "Warning", "Please enter a query.")
                self.progress_bar.setVisible(False)
                return
                
            data = {
                'query': query,
                'dataframe': self.current_data
            }
        elif operation == "batch_analysis":
            # Get user-defined queries
            queries_text = self.batch_queries_input.toPlainText().strip()
            if not queries_text:
                QMessageBox.warning(self, "Warning", "Please enter at least one query.")
                self.progress_bar.setVisible(False)
                return
                
            # Split by newlines and filter out empty lines
            queries = [q.strip() for q in queries_text.split('\n') if q.strip()]
            
            if not queries:
                QMessageBox.warning(self, "Warning", "Please enter at least one valid query.")
                self.progress_bar.setVisible(False)
                return
                
            data = {
                'queries': queries,
                'dataframe': self.current_data
            }
        else:
            # For other operations, just pass the dataframe
            data = self.current_data
        
        # Get API key
        current_api_key = self.api_key
        if not current_api_key and self.api_key_input.text() and not self.api_key_input.text().startswith("*"):
            current_api_key = self.api_key_input.text()
        
        # Start processing thread
        self.processing_thread = DataAnalysisThread(operation, data, current_api_key)
        self.processing_thread.update_progress.connect(self.update_progress)
        self.processing_thread.result_ready.connect(self.handle_results)
        self.processing_thread.error_occurred.connect(self.handle_error)
        self.processing_thread.start()
        
        # Disable run button while processing
        self.run_button.setEnabled(False)
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
    
    def handle_results(self, results):
        """Process and display the results from Claude analysis"""
        # Hide progress bar and re-enable run button
        self.progress_bar.setVisible(False)
        self.run_button.setEnabled(True)
        
        # Check for errors
        if "error" in results:
            self.handle_error(results["error"])
            return
            
        result_type = results.get("type", "unknown")
        print(f"Received results of type: {result_type}")  # Debug print
        
        # Display the main result in the text output
        if "result" in results:
            self.text_output.append(results["result"])
        
        # Create appropriate visualizations based on result type
        if result_type == "data_summary":
            self.visualize_data_summary(results)
        elif result_type == "experiment_analysis":
                self.visualize_experiment_analysis(results)
        elif result_type == "gene_analysis":
                self.visualize_gene_analysis(results)
        elif result_type == "phenotype_prediction":
                self.visualize_phenotype_prediction(results)
        elif result_type == "custom_query":
                self.visualize_custom_query(results)
        elif result_type == "batch_analysis":
                self.visualize_batch_analysis(results)
        else:
            print(f"No visualization implemented for result type: {result_type}")
    
    def run_batch_analysis(self):
        """Run multiple analyses in a single batch request"""
        self.update_progress.emit(10)
        
        if not isinstance(self.data, pd.DataFrame):
            return {"error": "Invalid data format. Expected DataFrame."}
        
        df = self.data
        
        # Get data context
        data_context = self.dataframe_to_context(df)
        
        # Define multiple analysis prompts
        summary_prompt = f"""Here is a dataset to summarize:
        
        {data_context}
        
        Please provide a brief overview of what this dataset contains and key observations."""
        
        patterns_prompt = f"""Here is a dataset to analyze:
        
        {data_context}
        
        Please identify the top 3 most interesting patterns or relationships in this data."""
        
        recommendations_prompt = f"""Here is a dataset to analyze:
        
        {data_context}
        
        Please provide 3 specific recommendations for further analysis or experiments based on this data."""
        
        self.update_progress.emit(20)
        
        # Process each prompt individually (safer approach without batch API)
        results = {}
        
        # Process summary
        message = self.anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=500,
            temperature=0.1,
            messages=[{"role": "user", "content": summary_prompt}]
        )
        results['summary-analysis'] = message.content[0].text
        self.update_progress.emit(50)
        
        # Process patterns
        message = self.anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=500,
            temperature=0.1,
            messages=[{"role": "user", "content": patterns_prompt}]
        )
        results['patterns-analysis'] = message.content[0].text
        self.update_progress.emit(70)
        
        # Process recommendations
        message = self.anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=500,
            temperature=0.1,
            messages=[{"role": "user", "content": recommendations_prompt}]
        )
        results['recommendations-analysis'] = message.content[0].text
        self.update_progress.emit(90)
        
        # Combine the results
        combined_analysis = f"""# Comprehensive Dataset Analysis
    
    ## Summary
    {results.get('summary-analysis', 'Analysis unavailable')}
    
    ## Key Patterns
    {results.get('patterns-analysis', 'Analysis unavailable')}
    
    ## Recommendations
    {results.get('recommendations-analysis', 'Analysis unavailable')}
    """
        
        self.update_progress.emit(100)
        
        return {
            "type": "batch_analysis",
            "result": combined_analysis,
            "individual_results": results,
            "metadata": {
                "num_records": len(df),
                "num_analyses": len(results),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
            
    def visualize_batch_analysis(self, results):
        """Create visualization for batch analysis results with user-defined queries"""
        individual_results = results.get("individual_results", {})
        
        # Create a simple visualization showing the different analysis components
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Create a bar chart showing the length of each analysis response
        components = []
        lengths = []
        labels = []
        
        for query_id, query_data in individual_results.items():
            components.append(query_id)
            query_text = query_data.get('query', query_id)
            # Truncate query text if it's too long
            if len(query_text) > 20:
                labels.append(query_text[:17] + "...")
            else:
                labels.append(query_text)
            lengths.append(len(query_data.get('result', '')))
        
        if components:
            bars = ax.bar(labels, lengths, color=['#5cb85c', '#d9534f', '#5bc0de', '#f0ad4e', '#337ab7'])
            
            # Add count labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                       f'{int(height)} chars', ha='center', va='bottom')
            
            ax.set_title('Response Length by Query')
            ax.set_ylabel('Character Count')
            
            # Rotate x-axis labels for readability
            plt.xticks(rotation=45, ha='right')
            
            self.figure.tight_layout()
            self.canvas.draw()
        
        # Create table view of batch analysis components
        self.results_table.clear()
        self.results_table.setRowCount(len(components))
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(["Query", "Preview"])
        
        for i, query_id in enumerate(components):
            query_data = individual_results[query_id]
            query_text = query_data.get('query', '')
            self.results_table.setItem(i, 0, QTableWidgetItem(query_text))
            
            # Add preview of the result (truncated if needed)
            content = query_data.get('result', '')
            if len(content) > 100:
                preview = content[:97] + "..."
            else:
                preview = content
            self.results_table.setItem(i, 1, QTableWidgetItem(preview))
        
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
    
    def handle_error(self, error_message):
        """Handle errors from Claude operations"""
        self.progress_bar.setVisible(False)
        self.run_button.setEnabled(True)
        self.text_output.setTextColor(Qt.red)
        self.text_output.append(f"Error: {error_message}")
        self.text_output.setTextColor(Qt.black)
    
    def visualize_data_summary(self, results):
        """Create visualization for data summary results"""
        metadata = results.get("metadata", {})
        
        # Create a simple visualization of the dataset structure
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Display columns by type if we have current_data
        if self.current_data is not None:
            df = self.current_data
            
            # Count column types
            numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
            categorical_cols = len(df.select_dtypes(include=['object']).columns)
            date_cols = len(df.select_dtypes(include=['datetime']).columns)
            other_cols = len(df.columns) - numeric_cols - categorical_cols - date_cols
            
            # Create bar chart of column types
            col_types = ['Numeric', 'Categorical', 'Date/Time', 'Other']
            counts = [numeric_cols, categorical_cols, date_cols, other_cols]
            
            # Filter out zero counts
            valid_types = [(t, c) for t, c in zip(col_types, counts) if c > 0]
            if valid_types:
                types, counts = zip(*valid_types)
                
                bars = ax.bar(types, counts, color=['#5cb85c', '#d9534f', '#5bc0de', '#f0ad4e'])
                
                # Add count labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{int(height)}', ha='center', va='bottom')
                
                ax.set_title('Column Types in Dataset')
                ax.set_ylabel('Count')
                ax.set_ylim(0, max(counts) * 1.2)  # Add some headroom for labels
                
                self.figure.tight_layout()
                self.canvas.draw()
        
        # Create table view of dataset info
        self.results_table.clear()
        self.results_table.setRowCount(len(self.current_data.columns))
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["Column", "Type", "Missing Values", "Sample Values"])
        
        for i, col in enumerate(self.current_data.columns):
            # Column name
            self.results_table.setItem(i, 0, QTableWidgetItem(col))
            
            # Data type
            dtype = str(self.current_data[col].dtype)
            self.results_table.setItem(i, 1, QTableWidgetItem(dtype))
            
            # Missing values
            missing = self.current_data[col].isna().sum()
            missing_pct = 100 * missing / len(self.current_data)
            self.results_table.setItem(i, 2, QTableWidgetItem(f"{missing} ({missing_pct:.1f}%)"))
            
            # Sample values
            if self.current_data[col].dtype == 'object':
                sample = ', '.join(str(x) for x in self.current_data[col].dropna().unique()[:3])
                if len(self.current_data[col].dropna().unique()) > 3:
                    sample += ", ..."
            else:
                sample = f"min: {self.current_data[col].min()}, max: {self.current_data[col].max()}"
            
            self.results_table.setItem(i, 3, QTableWidgetItem(sample))
        
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.results_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
    
    def visualize_experiment_analysis(self, results):
        """Create visualization for experiment analysis results"""
        # Extract keywords for word frequency analysis
        result_text = results.get("result", "")
        
        # Create a word frequency count (basic NLP visualization)
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Extract words and count frequencies
        words = re.findall(r'\b[A-Za-z]{3,15}\b', result_text.lower())
        
        # Remove common stop words
        stop_words = {'the', 'and', 'is', 'in', 'to', 'of', 'that', 'this', 'for', 'with', 
                     'are', 'be', 'on', 'as', 'by', 'they', 'was', 'can', 'from', 'have', 'has'}
        filtered_words = [word for word in words if word not in stop_words]
        
        # Count frequencies
        from collections import Counter
        word_counts = Counter(filtered_words)
        
        # Get top words
        top_words = word_counts.most_common(10)
        
        if top_words:
            words, counts = zip(*top_words)
            
            # Create horizontal bar chart
            bars = ax.barh(list(reversed(words)), list(reversed(counts)), color='skyblue')
            
            # Add count labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.3, bar.get_y() + bar.get_height()/2.,
                       f'{int(width)}', ha='left', va='center')
            
            ax.set_title('Key Terms Frequency')
            ax.set_xlabel('Frequency')
            
            self.figure.tight_layout()
            self.canvas.draw()
        
        # Create a table view with key findings
        # Extract key insights from the result text (looking for patterns like numbered lists)
        insights = re.findall(r'\d+\.\s+(.*?)(?=\d+\.\s+|\Z)', result_text, re.DOTALL)
        key_findings = re.findall(r'(?:key\s+finding|conclusion|insight|discovery)[:\s]+(.*?)(?=\n\n|\Z)', 
                                 result_text, re.IGNORECASE | re.DOTALL)
        
        all_insights = insights + key_findings
        
        if all_insights:
            self.results_table.clear()
            self.results_table.setRowCount(min(len(all_insights), 10))  # Limit to 10 rows
            self.results_table.setColumnCount(1)
            self.results_table.setHorizontalHeaderLabels(["Key Insights"])
            
            for i, insight in enumerate(all_insights[:10]):
                # Clean up text
                clean_insight = re.sub(r'\s+', ' ', insight).strip()
                self.results_table.setItem(i, 0, QTableWidgetItem(clean_insight))
            
            self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
    
    def visualize_gene_analysis(self, results):
        """Create visualization for gene analysis results"""
        clusters = results.get("clusters", {})
        
        if clusters:
            # Create a visualization of gene clusters
            self.figure.clear()
            
            # Create a grid based on the number of clusters
            n_clusters = len(clusters)
            n_cols = min(3, n_clusters)
            n_rows = (n_clusters + n_cols - 1) // n_cols
            
            # Set up the figure to have multiple subplots
            for i, (cluster_name, cluster_data) in enumerate(clusters.items()):
                # Create subplot
                ax = self.figure.add_subplot(n_rows, n_cols, i+1)
                
                genes = cluster_data.get('genes', [])
                if genes:
                    # Create bar chart of gene counts
                    y_pos = range(len(genes))
                    counts = [1] * len(genes)  # All genes have equal weight
                    ax.barh(y_pos, counts, color='lightblue')
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(genes)
                    ax.set_title(cluster_name)
                    ax.set_xlim(0, 1.2)  # Fixed scale
                    ax.set_xticks([])  # Hide x-axis
                    
                    # Line wrap for description
                    desc = cluster_data.get('description', '')
                    if len(desc) > 30:
                        desc = desc[:27] + "..."
                    
                    # Add shortened description at the bottom
                    ax.text(0.5, -0.1, desc, transform=ax.transAxes,
                           ha='center', va='top', fontsize=8)
            
            self.figure.tight_layout()
            self.canvas.draw()
            
            # Create table view of clusters
            self.results_table.clear()
            self.results_table.setRowCount(sum(len(c.get('genes', [])) for c in clusters.values()))
            self.results_table.setColumnCount(2)
            self.results_table.setHorizontalHeaderLabels(["Cluster", "Gene"])
            
            row = 0
            for cluster_name, cluster_data in clusters.items():
                genes = cluster_data.get('genes', [])
                for gene in genes:
                    self.results_table.setItem(row, 0, QTableWidgetItem(cluster_name))
                    self.results_table.setItem(row, 1, QTableWidgetItem(gene))
                    row += 1
            
            self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
            self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
    
    def visualize_phenotype_prediction(self, results):
        """Create visualization for phenotype prediction results"""
        result_text = results.get("result", "")
        
        # Try to extract structured phenotype predictions
        phenotype_categories = ["eye", "head", "pericardium", "yolk", "larva", "tail"]
        
        phenotype_data = {}
        for category in phenotype_categories:
            # Look for sections that discuss this phenotype
            pattern = rf'(?:{category}|{category.capitalize()})[:\s]+(.*?)(?=\n\n|\Z)'
            matches = re.findall(pattern, result_text, re.IGNORECASE | re.DOTALL)
            
            if matches:
                phenotype_data[category] = matches[0].strip()
        
        # Create visualization of phenotype predictions
        self.figure.clear()
        
        if phenotype_data:
            # Create radar chart of phenotype predictions
            ax = self.figure.add_subplot(111, polar=True)
            
            # Count number of words in each phenotype description as a simple metric
            categories = list(phenotype_data.keys())
            values = [len(phenotype_data[cat].split()) for cat in categories]
            
            # Complete the loop
            categories.append(categories[0])
            values.append(values[0])
            
            # Plot radar chart
            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=True)
            ax.plot(angles, values, 'o-', linewidth=2)
            ax.fill(angles, values, alpha=0.25)
            ax.set_thetagrids(angles[:-1] * 180/np.pi, categories[:-1])
            ax.set_title('Phenotype Prediction Detail Level')
            ax.grid(True)
            
            self.figure.tight_layout()
            self.canvas.draw()
        else:
            # Alternative: Create word frequency visualization
            words = re.findall(r'\b[A-Za-z]{3,15}\b', result_text.lower())
            
            # Filter out common words
            stop_words = {'the', 'and', 'is', 'in', 'to', 'of', 'that', 'this', 'for', 'with'}
            filtered_words = [word for word in words if word not in stop_words]
            
            # Count frequencies
            from collections import Counter
            word_counts = Counter(filtered_words)
            
            # Get top phenotype-related words
            phenotype_related = ["wild", "type", "mutant", "normal", "abnormal", "defect", 
                               "malformation", "reduced", "increased", "altered",
                               "eye", "head", "pericardium", "yolk", "larva", "tail"]
                               
            top_terms = {}
            for term in phenotype_related:
                if term in word_counts:
                    top_terms[term] = word_counts[term]
            
            # If we don't have enough phenotype terms, add other top words
            if len(top_terms) < 8:
                for word, count in word_counts.most_common(10):
                    if word not in top_terms and len(top_terms) < 8:
                        top_terms[word] = count
            
            if top_terms:
                # Create bar chart
                ax = self.figure.add_subplot(111)
                words, counts = zip(*top_terms.items())
                
                bars = ax.bar(words, counts, color='lightgreen')
                
                # Add count labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{int(height)}', ha='center', va='bottom')
                
                ax.set_title('Key Phenotype Terms Frequency')
                ax.set_ylabel('Frequency')
                
                # Rotate x-axis labels for readability
                plt.xticks(rotation=45, ha='right')
                
                self.figure.tight_layout()
                self.canvas.draw()
        
        # Create table view of phenotype predictions
        self.results_table.clear()
        
        # Try to extract predictions with confidence levels
        confidence_pattern = r'(\b(?:high|medium|low)\s+confidence\b)'
        confidence_matches = re.findall(confidence_pattern, result_text, re.IGNORECASE)
        
        # If we found confidence levels, create a structured table
        if confidence_matches:
            # Look for patterns like "Phenotype X: Description (high confidence)"
            pattern = r'([A-Za-z\s]+):\s*([^(]+)\s*\(([^)]+)\)'
            predictions = re.findall(pattern, result_text)
            
            if predictions:
                self.results_table.setRowCount(len(predictions))
                self.results_table.setColumnCount(3)
                self.results_table.setHorizontalHeaderLabels(["Phenotype", "Description", "Confidence"])
                
                for i, (phenotype, description, confidence) in enumerate(predictions):
                    self.results_table.setItem(i, 0, QTableWidgetItem(phenotype.strip()))
                    self.results_table.setItem(i, 1, QTableWidgetItem(description.strip()))
                    self.results_table.setItem(i, 2, QTableWidgetItem(confidence.strip()))
            else:
                # Use the phenotype categories we extracted earlier
                self.results_table.setRowCount(len(phenotype_data))
                self.results_table.setColumnCount(2)
                self.results_table.setHorizontalHeaderLabels(["Phenotype", "Prediction"])
                
                for i, (phenotype, prediction) in enumerate(phenotype_data.items()):
                    self.results_table.setItem(i, 0, QTableWidgetItem(phenotype.capitalize()))
                    self.results_table.setItem(i, 1, QTableWidgetItem(prediction))
        else:
            # Use the phenotype categories we extracted earlier
            self.results_table.setRowCount(len(phenotype_data))
            self.results_table.setColumnCount(2)
            self.results_table.setHorizontalHeaderLabels(["Phenotype", "Prediction"])
            
            for i, (phenotype, prediction) in enumerate(phenotype_data.items()):
                self.results_table.setItem(i, 0, QTableWidgetItem(phenotype.capitalize()))
                self.results_table.setItem(i, 1, QTableWidgetItem(prediction))
        
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
    
    def visualize_custom_query(self, results):
        """Create visualization for custom query results"""
        query = results.get("query", "")
        result_text = results.get("result", "")
        
        # Try to create a word cloud visualization
        try:
            from wordcloud import WordCloud
            
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            # Generate word cloud
            wordcloud = WordCloud(width=800, height=400, 
                                background_color='white',
                                max_words=100, 
                                contour_width=1, 
                                contour_color='steelblue').generate(result_text)
            
            # Display the word cloud
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.set_title("Response Word Cloud")
            ax.axis('off')
            
            self.figure.tight_layout()
            self.canvas.draw()
            
        except ImportError:
            # If wordcloud not available, show a simple text visualization
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            # Extract key terms (non-stop words with 5+ characters)
            words = re.findall(r'\b[A-Za-z]{5,15}\b', result_text.lower())
            
            # Filter out common scientific paper words
            stop_words = {'about', 'above', 'after', 'again', 'against', 'because', 
                        'before', 'being', 'between', 'both', 'cannot', 'could', 
                        'during', 'each', 'either', 'might', 'other', 'should', 
                        'their', 'there', 'these', 'those', 'through', 'under', 
                        'where', 'which', 'while', 'would', 'analysis', 'study',
                        'research', 'results', 'based', 'using', 'different', 'however'}
            
            filtered_words = [word for word in words if word not in stop_words]
            
            # Count frequencies
            from collections import Counter
            word_counts = Counter(filtered_words)
            
            # Get top words
            top_words = word_counts.most_common(10)
            
            if top_words:
                words, counts = zip(*top_words)
                
                # Create horizontal bar chart
                y_pos = np.arange(len(words))
                ax.barh(y_pos, counts, align='center', color='skyblue')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(words)
                ax.invert_yaxis()  # labels read top-to-bottom
                ax.set_title('Key Terms in Response')
                ax.set_xlabel('Frequency')
                
                # Add count labels
                for i, v in enumerate(counts):
                    ax.text(v + 0.1, i, str(v), va='center')
                
                self.figure.tight_layout()
                self.canvas.draw()
        
        # Create a table view of key points from the response
        # Try to extract key statements or bullet points
        
        # Look for bullet points or numbered lists
        bullet_pattern = r'(?:•|\*|\d+\.)\s+([^\n]+)'
        bullets = re.findall(bullet_pattern, result_text)
        
        if bullets:
            self.results_table.clear()
            self.results_table.setRowCount(len(bullets))
            self.results_table.setColumnCount(1)
            self.results_table.setHorizontalHeaderLabels(["Key Points"])
            
            for i, bullet in enumerate(bullets):
                self.results_table.setItem(i, 0, QTableWidgetItem(bullet))
            
            self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        else:
            # Split by paragraphs and use those if no bullet points found
            paragraphs = [p.strip() for p in result_text.split('\n\n') if p.strip()]
            
            if paragraphs:
                self.results_table.clear()
                self.results_table.setRowCount(min(len(paragraphs), 10))
                self.results_table.setColumnCount(1)
                self.results_table.setHorizontalHeaderLabels(["Key Paragraphs"])
                
                for i, paragraph in enumerate(paragraphs[:10]):  # Limit to 10
                    # Truncate very long paragraphs
                    if len(paragraph) > 200:
                        paragraph = paragraph[:197] + "..."
                    self.results_table.setItem(i, 0, QTableWidgetItem(paragraph))
                
                self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    analysis_widget = ClaudeAnalysis()
    analysis_widget.show()
    sys.exit(app.exec_())