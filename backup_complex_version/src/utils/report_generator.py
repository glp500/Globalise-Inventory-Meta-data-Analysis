"""
Report generation utilities for VOC classification results.
"""

from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import json


logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates detailed reports and visualizations for classification results."""
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_summary_report(self, results: List) -> Dict[str, Any]:
        """
        Generate a comprehensive summary report.
        
        Args:
            results: List of classification results
            
        Returns:
            Summary statistics dictionary
        """
        if not results:
            return {}
        
        # Convert ClassificationResult objects to dictionaries first
        results_dicts = []
        for result in results:
            if hasattr(result, 'image_path'):
                result_dict = {
                    'image_path': result.image_path,
                    'category': result.category,
                    'confidence': result.confidence,
                    'page_type': result.page_type,
                    'features': result.features,
                    'timestamp': result.timestamp,
                    'error': result.error
                }
            else:
                result_dict = result
            results_dicts.append(result_dict)
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(results_dicts)
        
        # Basic statistics
        total_files = len(results)
        categories = df['category'].value_counts().to_dict()
        avg_confidence = df['confidence'].mean() if 'confidence' in df.columns else 0.0
        
        # Confidence distribution
        confidence_stats = {}
        if 'confidence' in df.columns:
            confidence_stats = {
                'mean': df['confidence'].mean(),
                'median': df['confidence'].median(),
                'std': df['confidence'].std(),
                'min': df['confidence'].min(),
                'max': df['confidence'].max()
            }
        
        # Error analysis
        errors = df[df['error'].notna()] if 'error' in df.columns else pd.DataFrame()
        error_count = len(errors)
        error_rate = error_count / total_files if total_files > 0 else 0
        
        # Page type distribution
        page_types = df['page_type'].value_counts().to_dict() if 'page_type' in df.columns else {}
        
        summary = {
            'processing_summary': {
                'total_files': total_files,
                'successful_classifications': total_files - error_count,
                'errors': error_count,
                'error_rate': error_rate,
                'average_confidence': avg_confidence
            },
            'category_distribution': categories,
            'page_type_distribution': page_types,
            'confidence_statistics': confidence_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        return summary
    
    def create_visualization_report(self, results: List, save_plots: bool = True) -> Optional[Path]:
        """
        Create visualization report with charts and graphs.
        
        Args:
            results: List of classification results
            save_plots: Whether to save plots to files
            
        Returns:
            Path to saved report or None
        """
        if not results:
            logger.warning("No results to visualize")
            return None
        
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('VOC Document Classification Report', fontsize=16, fontweight='bold')
            
            # Convert ClassificationResult objects to dictionaries first
            results_dicts = []
            for result in results:
                if hasattr(result, 'image_path'):
                    result_dict = {
                        'image_path': result.image_path,
                        'category': result.category,
                        'confidence': result.confidence,
                        'page_type': result.page_type,
                        'features': result.features,
                        'timestamp': result.timestamp,
                        'error': result.error
                    }
                else:
                    result_dict = result
                results_dicts.append(result_dict)
            
            df = pd.DataFrame(results_dicts)
            
            # 1. Category distribution pie chart
            if 'category' in df.columns:
                category_counts = df['category'].value_counts()
                axes[0, 0].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
                axes[0, 0].set_title('Category Distribution')
            
            # 2. Confidence score histogram
            if 'confidence' in df.columns:
                axes[0, 1].hist(df['confidence'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                axes[0, 1].set_title('Confidence Score Distribution')
                axes[0, 1].set_xlabel('Confidence Score')
                axes[0, 1].set_ylabel('Frequency')
            
            # 3. Page type distribution bar chart
            if 'page_type' in df.columns:
                page_type_counts = df['page_type'].value_counts()
                axes[1, 0].bar(page_type_counts.index, page_type_counts.values, color='lightcoral')
                axes[1, 0].set_title('Page Type Distribution')
                axes[1, 0].set_xlabel('Page Type')
                axes[1, 0].set_ylabel('Count')
                axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 4. Processing status summary
            if 'error' in df.columns:
                success_count = df['error'].isna().sum()
                error_count = df['error'].notna().sum()
                
                labels = ['Successful', 'Errors']
                sizes = [success_count, error_count]
                colors = ['lightgreen', 'lightcoral']
                
                axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
                axes[1, 1].set_title('Processing Status')
            
            plt.tight_layout()
            
            if save_plots:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_path = self.output_dir / f"classification_visualization_{timestamp}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved visualization report to: {plot_path}")
                return plot_path
            
            plt.show()
            return None
            
        except Exception as e:
            logger.error(f"Error creating visualization report: {e}")
            return None
    
    def generate_detailed_report(self, results: List) -> Path:
        """
        Generate detailed HTML report with all analysis.
        
        Args:
            results: List of classification results
            
        Returns:
            Path to generated HTML report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"detailed_report_{timestamp}.html"
        
        summary = self.generate_summary_report(results)
        
        html_content = self._generate_html_report(summary, results)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Generated detailed report: {report_path}")
        return report_path
    
    def _generate_html_report(self, summary: Dict[str, Any], results: List[Dict[str, Any]]) -> str:
        """Generate HTML content for detailed report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>VOC Document Classification Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .stat-box {{ 
                    display: inline-block; 
                    padding: 10px; 
                    margin: 5px; 
                    background-color: #e6f3ff; 
                    border-radius: 5px; 
                }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .error {{ color: red; }}
                .success {{ color: green; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>VOC Document Classification Report</h1>
                <p>Generated on: {summary.get('timestamp', 'Unknown')}</p>
            </div>
            
            <div class="section">
                <h2>Processing Summary</h2>
                {self._format_processing_summary(summary.get('processing_summary', {}))}
            </div>
            
            <div class="section">
                <h2>Category Distribution</h2>
                {self._format_category_table(summary.get('category_distribution', {}))}
            </div>
            
            <div class="section">
                <h2>Confidence Statistics</h2>
                {self._format_confidence_stats(summary.get('confidence_statistics', {}))}
            </div>
            
            <div class="section">
                <h2>Detailed Results</h2>
                {self._format_results_table(results[:100])}  <!-- Limit to first 100 for performance -->
                {f"<p><em>Showing first 100 of {len(results)} results</em></p>" if len(results) > 100 else ""}
            </div>
        </body>
        </html>
        """
        return html
    
    def _format_processing_summary(self, summary: Dict[str, Any]) -> str:
        """Format processing summary as HTML."""
        return f"""
        <div class="stat-box">Total Files: <strong>{summary.get('total_files', 0)}</strong></div>
        <div class="stat-box">Successful: <strong class="success">{summary.get('successful_classifications', 0)}</strong></div>
        <div class="stat-box">Errors: <strong class="error">{summary.get('errors', 0)}</strong></div>
        <div class="stat-box">Error Rate: <strong>{summary.get('error_rate', 0):.2%}</strong></div>
        <div class="stat-box">Avg Confidence: <strong>{summary.get('average_confidence', 0):.3f}</strong></div>
        """
    
    def _format_category_table(self, categories: Dict[str, int]) -> str:
        """Format category distribution as HTML table."""
        if not categories:
            return "<p>No category data available</p>"
        
        rows = ""
        for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / sum(categories.values())) * 100
            rows += f"<tr><td>{category}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>"
        
        return f"""
        <table>
            <thead>
                <tr><th>Category</th><th>Count</th><th>Percentage</th></tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
        """
    
    def _format_confidence_stats(self, stats: Dict[str, float]) -> str:
        """Format confidence statistics as HTML."""
        if not stats:
            return "<p>No confidence data available</p>"
        
        return f"""
        <div class="stat-box">Mean: <strong>{stats.get('mean', 0):.3f}</strong></div>
        <div class="stat-box">Median: <strong>{stats.get('median', 0):.3f}</strong></div>
        <div class="stat-box">Std Dev: <strong>{stats.get('std', 0):.3f}</strong></div>
        <div class="stat-box">Min: <strong>{stats.get('min', 0):.3f}</strong></div>
        <div class="stat-box">Max: <strong>{stats.get('max', 0):.3f}</strong></div>
        """
    
    def _format_results_table(self, results: List) -> str:
        """Format results as HTML table."""
        if not results:
            return "<p>No results to display</p>"
        
        rows = ""
        for result in results:
            if hasattr(result, 'image_path'):
                filename = Path(result.image_path).name
                category = result.category
                confidence = result.confidence
                page_type = result.page_type
                error = result.error or ''
            else:
                filename = Path(result.get('image_path', '')).name
                category = result.get('category', 'Unknown')
                confidence = result.get('confidence', 0)
                page_type = result.get('page_type', 'Unknown')
                error = result.get('error', '')
            
            error_class = 'error' if error else ''
            rows += f"""
            <tr class="{error_class}">
                <td>{filename}</td>
                <td>{category}</td>
                <td>{confidence:.3f}</td>
                <td>{page_type}</td>
                <td>{error if error else 'Success'}</td>
            </tr>
            """
        
        return f"""
        <table>
            <thead>
                <tr>
                    <th>Filename</th>
                    <th>Category</th>
                    <th>Confidence</th>
                    <th>Page Type</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
        """
    
    def export_for_analysis(self, results: List, format: str = 'excel') -> Path:
        """
        Export results in format suitable for further analysis.
        
        Args:
            results: Classification results
            format: Export format ('excel', 'csv', 'json')
            
        Returns:
            Path to exported file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format.lower() == 'excel':
            export_path = self.output_dir / f"analysis_data_{timestamp}.xlsx"
            # Convert ClassificationResult objects to dictionaries first
            results_dicts = []
            for result in results:
                if hasattr(result, 'image_path'):
                    result_dict = {
                        'image_path': result.image_path,
                        'category': result.category,
                        'confidence': result.confidence,
                        'page_type': result.page_type,
                        'features': result.features,
                        'timestamp': result.timestamp,
                        'error': result.error
                    }
                else:
                    result_dict = result
                results_dicts.append(result_dict)
            
            df = pd.DataFrame(results_dicts)
            
            with pd.ExcelWriter(export_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Results', index=False)
                
                # Add summary sheet
                summary = self.generate_summary_report(results)
                summary_df = pd.DataFrame([summary['processing_summary']])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
        elif format.lower() == 'csv':
            export_path = self.output_dir / f"analysis_data_{timestamp}.csv"
            # Convert ClassificationResult objects to dictionaries first
            results_dicts = []
            for result in results:
                if hasattr(result, 'image_path'):
                    result_dict = {
                        'image_path': result.image_path,
                        'category': result.category,
                        'confidence': result.confidence,
                        'page_type': result.page_type,
                        'features': result.features,
                        'timestamp': result.timestamp,
                        'error': result.error
                    }
                else:
                    result_dict = result
                results_dicts.append(result_dict)
            
            df = pd.DataFrame(results_dicts)
            df.to_csv(export_path, index=False)
            
        elif format.lower() == 'json':
            export_path = self.output_dir / f"analysis_data_{timestamp}.json"
            # Convert ClassificationResult objects to dictionaries first
            results_dicts = []
            for result in results:
                if hasattr(result, 'image_path'):
                    result_dict = {
                        'image_path': result.image_path,
                        'category': result.category,
                        'confidence': result.confidence,
                        'page_type': result.page_type,
                        'features': result.features,
                        'timestamp': result.timestamp,
                        'error': result.error
                    }
                else:
                    result_dict = result
                results_dicts.append(result_dict)
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(results_dicts, f, indent=2, ensure_ascii=False)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported analysis data to: {export_path}")
        return export_path