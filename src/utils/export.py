"""Export utilities for analysis results"""
import json
import os
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

from fpdf import FPDF
import markdown


class PDFExport(FPDF):
    """Custom PDF export class"""
    
    def header(self):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, 'Research Paper Analysis Report', 0, 1, 'C')
        self.line(10, 20, 200, 20)
        self.ln(5)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')
    
    def section_title(self, title: str):
        self.set_font('Helvetica', 'B', 14)
        self.set_fill_color(230, 230, 230)
        self.cell(0, 10, title, 0, 1, 'L', fill=True)
        self.ln(3)
    
    def body_text(self, text: str):
        self.set_font('Helvetica', '', 11)
        self.multi_cell(0, 6, text)
        self.ln(3)
    
    def bullet_point(self, text: str):
        self.set_font('Helvetica', '', 11)
        x = self.get_x()
        self.cell(5, 6, '-')
        self.multi_cell(0, 6, f'  {text}')
        self.ln(1)


def export_to_json(analysis_dict: Dict[str, Any], output_path: str) -> str:
    """Export analysis to JSON file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_dict, f, indent=2, ensure_ascii=False)
    return output_path


def export_to_markdown(analysis_dict: Dict[str, Any], output_path: str) -> str:
    """Export analysis to Markdown file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    md_lines = [
        f"# {analysis_dict['paper']['title']}",
        "",
        f"**Authors:** {', '.join(analysis_dict['paper']['authors']) if analysis_dict['paper']['authors'] else 'Unknown'}",
        f"**Pages:** {analysis_dict['paper']['page_count']}",
        f"**Analyzed:** {analysis_dict['analyzed_at']}",
        "",
        "## Abstract",
        analysis_dict['paper']['abstract'],
        "",
        "## Overall Summary",
        analysis_dict['overall_summary']['summary'],
        "",
        "### Key Points",
    ]
    
    for point in analysis_dict['overall_summary']['key_points']:
        md_lines.append(f"- {point}")
    
    md_lines.extend(["", "## Section Summaries"])
    
    for section_title, summary in analysis_dict['section_summaries'].items():
        md_lines.extend([
            f"### {section_title}",
            summary['summary'],
            "",
            "**Key Points:**",
        ])
        for point in summary['key_points']:
            md_lines.append(f"- {point}")
        md_lines.append("")
    
    md_lines.extend([
        "## Key Terms",
        ", ".join(analysis_dict['key_terms']),
        "",
        "## Readability Scores",
    ])
    
    for metric, score in analysis_dict['readability'].items():
        md_lines.append(f"- **{metric.replace('_', ' ').title()}:** {score:.2f}")
    
    md_lines.extend([
        "",
        "## Statistics",
    ])
    
    for key, value in analysis_dict['statistics'].items():
        md_lines.append(f"- **{key.replace('_', ' ').title()}:** {value}")
    
    content = "\n".join(md_lines)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return output_path


def export_to_pdf(analysis_dict: Dict[str, Any], output_path: str) -> str:
    """Export analysis to PDF file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    pdf = PDFExport()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)
    
    # Title
    pdf.set_font('Helvetica', 'B', 18)
    pdf.cell(0, 15, analysis_dict['paper']['title'], 0, 1, 'C')
    pdf.ln(5)
    
    # Metadata
    pdf.set_font('Helvetica', '', 11)
    if analysis_dict['paper']['authors']:
        pdf.cell(0, 7, f"Authors: {', '.join(analysis_dict['paper']['authors'])}", 0, 1)
    pdf.cell(0, 7, f"Pages: {analysis_dict['paper']['page_count']}", 0, 1)
    pdf.cell(0, 7, f"Analyzed: {analysis_dict['analyzed_at'][:10]}", 0, 1)
    pdf.ln(5)
    
    # Abstract
    pdf.section_title("Abstract")
    pdf.body_text(analysis_dict['paper']['abstract'])
    
    # Overall Summary
    pdf.section_title("Overall Summary")
    pdf.body_text(analysis_dict['overall_summary']['summary'])
    
    pdf.section_title("Key Points")
    for point in analysis_dict['overall_summary']['key_points']:
        pdf.bullet_point(point)
    
    # Section Summaries
    if analysis_dict['section_summaries']:
        pdf.section_title("Section Summaries")
        for section_title, summary in analysis_dict['section_summaries'].items():
            pdf.section_title(section_title)
            pdf.body_text(summary['summary'])
            for point in summary['key_points']:
                pdf.bullet_point(point)
    
    # Key Terms
    pdf.section_title("Key Terms")
    pdf.body_text(", ".join(analysis_dict['key_terms']))
    
    # Readability
    if analysis_dict['readability']:
        pdf.section_title("Readability Scores")
        for metric, score in analysis_dict['readability'].items():
            pdf.body_text(f"{metric.replace('_', ' ').title()}: {score:.2f}")
    
    # Statistics
    pdf.section_title("Statistics")
    for key, value in analysis_dict['statistics'].items():
        pdf.body_text(f"{key.replace('_', ' ').title()}: {value}")
    
    pdf.output(output_path)
    return output_path


def export_analysis(analysis, format: str = "json", output_dir: str = None) -> str:
    """Export analysis in specified format"""
    if output_dir is None:
        output_dir = config.EXPORT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paper_title = analysis.paper.title[:50].replace(" ", "_").replace("/", "_")
    base_name = f"{paper_title}_{timestamp}"
    
    analysis_dict = analysis.to_dict()
    
    if format == "json":
        output_path = os.path.join(output_dir, f"{base_name}.json")
        return export_to_json(analysis_dict, output_path)
    elif format == "markdown" or format == "md":
        output_path = os.path.join(output_dir, f"{base_name}.md")
        return export_to_markdown(analysis_dict, output_path)
    elif format == "pdf":
        output_path = os.path.join(output_dir, f"{base_name}.pdf")
        return export_to_pdf(analysis_dict, output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")


# Import config here to avoid circular imports
from config.settings import config