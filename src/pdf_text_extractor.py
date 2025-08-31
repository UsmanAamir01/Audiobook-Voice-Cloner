#!/usr/bin/env python3
"""
PDF Text Extractor
=================
Extracts text from PDF and saves it in structured format for a        ]
        
        sections = {}
        text_lines = text.split('\n')
        current_section = "Introduction"
        current_text = ""
        section_count = 0k generation.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List
import logging

# PDF processing
try:
    import PyPDF2
except ImportError as e:
    raise ImportError("PyPDF2 is required. Please install it with: pip install PyPDF2") from e

try:
    import fitz  # PyMuPDF - better text extraction
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

class PDFTextExtractor:
    """Extract and structure text from PDF files"""
    
    def __init__(self, output_dir: str = "extracted_text"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'extraction.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def extract_text_pymupdf(self, pdf_path: str) -> str:
        """Extract text using PyMuPDF (better quality)"""
        text = ""
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            text += page_text + "\n"
            
            if (page_num + 1) % 10 == 0:
                self.logger.info(f"Processed {page_num + 1}/{len(doc)} pages")
        
        doc.close()
        return text
    
    def extract_text_pypdf2(self, pdf_path: str) -> str:
        """Extract text using PyPDF2 (fallback)"""
        text = ""
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                
                if page_num % 10 == 0:
                    self.logger.info(f"Processed {page_num}/{total_pages} pages")
        
        return text
    
    def extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF using best available method"""
        self.logger.info(f"Extracting text from: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            if PYMUPDF_AVAILABLE:
                self.logger.info("Using PyMuPDF for extraction...")
                text = self.extract_text_pymupdf(pdf_path)
            else:
                self.logger.info("Using PyPDF2 for extraction...")
                text = self.extract_text_pypdf2(pdf_path)
            
            self.logger.info(f"Extracted {len(text)} characters from PDF")
            return text
            
        except Exception as e:
            self.logger.error(f"Error extracting PDF: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers (common patterns)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        text = re.sub(r'\n\s*Page \d+.*?\n', '\n', text)
        
        # Remove special characters that cause TTS issues
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\"\'\(\)\[\]]', ' ', text)
        
        # Fix common spacing issues
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.,!?;:])\s*', r'\1 ', text)
        
        return text.strip()
    
    def detect_sections(self, text: str) -> Dict[str, str]:
        """Detect sections or parts in the text"""
        self.logger.info("Detecting sections and parts...")
        
        # Section detection patterns (ordered by priority)
        patterns = [
            r'(?:^|\n)\s*Section\s+(\d+)[:\.]?\s*([^\n]*)',
            r'(?:^|\n)\s*SECTION\s+(\d+)[:\.]?\s*([^\n]*)',
            r'(?:^|\n)\s*(\d+\.)\s+([A-Z][A-Za-z\s]+?)(?=\n|$)',
            r'(?:^|\n)\s*Section\s+(\d+)[:\.]?\s*([^\n]*)',
            r'(?:^|\n)\s*SECTION\s+(\d+)[:\.]?\s*([^\n]*)',
            r'(?:^|\n)\s*Part\s+(\d+)[:\.]?\s*([^\n]*)',
            r'(?:^|\n)\s*([A-Z][A-Z\s]{10,})\s*(?=\n)',  # ALL CAPS headings
        ]
        
        sections = {}
        lines = text.split('\n')
        current_section = "Introduction"
        current_text = ""
        section_count = 0
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            is_section = False
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE | re.MULTILINE)
                if match:
                    # Save previous section if it has content
                    if current_text.strip() and len(current_text.strip()) > 100:
                        sections[current_section] = self.clean_text(current_text.strip())
                        section_count += 1
                    
                    # Start new section
                    if len(match.groups()) >= 2:
                        section_num = match.group(1)
                        section_title = match.group(2).strip()
                        current_section = f"Section {section_num}: {section_title}" if section_title else f"Section {section_num}"
                    else:
                        current_section = match.group(1).strip() if match.groups() else line.strip()
                    
                    current_text = ""
                    is_section = True
                    self.logger.info(f"Found section: {current_section}")
                    break
            
            if not is_section:
                current_text += line + " "
        
        # Add last section
        if current_text.strip() and len(current_text.strip()) > 100:
            sections[current_section] = self.clean_text(current_text.strip())
            section_count += 1
        
        # If no proper sections found, create sections based on length
        if section_count < 2:
            self.logger.info("No clear sections found, creating sections by length...")
            sections = self.create_sections_by_length(text)
        
        self.logger.info(f"Detected {len(sections)} sections")
        return sections
    
    def create_sections_by_length(self, text: str, target_length: int = 3000) -> Dict[str, str]:
        """Create sections based on text length"""
        sections = {}
        words = text.split()
        
        section_num = 1
        current_section = []
        current_length = 0
        
        for word in words:
            current_section.append(word)
            current_length += len(word) + 1
            
            # Check for natural break points
            if current_length > target_length:
                # Look for sentence ending
                section_text = " ".join(current_section)
                sentences = re.split(r'[.!?]+', section_text)
                
                if len(sentences) > 1:
                    # Take all but the last incomplete sentence
                    complete_text = ". ".join(sentences[:-1]) + "."
                    remaining_words = sentences[-1].split()
                    
                    section_title = f"Section {section_num}"
                    sections[section_title] = self.clean_text(complete_text)
                    
                    # Start next section with remaining words
                    current_section = remaining_words
                    current_length = sum(len(w) + 1 for w in remaining_words)
                    section_num += 1
                else:
                    # No good break point, just split here
                    section_title = f"Section {section_num}"
                    sections[section_title] = self.clean_text(" ".join(current_section))
                    
                    current_section = []
                    current_length = 0
                    section_num += 1
        
        # Add remaining text
        if current_section:
            section_title = f"Section {section_num}"
            sections[section_title] = self.clean_text(" ".join(current_section))
        
        return sections
    
    def save_structured_text(self, sections: Dict[str, str], output_name: str) -> str:
        """Save structured text to JSON file"""
        output_file = self.output_dir / f"{output_name}_structured.json"
        
        structured_data = {
            "title": output_name,
            "extraction_date": str(Path().cwd()),
            "total_sections": len(sections),
            "sections": []
        }
        
        for i, (title, content) in enumerate(sections.items(), 1):
            section_data = {
                "number": i,
                "title": title,
                "content": content,
                "word_count": len(content.split()),
                "char_count": len(content)
            }
            structured_data["sections"].append(section_data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(structured_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved structured text to: {output_file}")
        return str(output_file)
    
    def save_plain_text(self, sections: Dict[str, str], output_name: str) -> str:
        """Save as plain text file"""
        output_file = self.output_dir / f"{output_name}_full_text.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, (title, content) in enumerate(sections.items(), 1):
                f.write(f"\n{'='*60}\n")
                f.write(f"CHAPTER {i}: {title.upper()}\n")
                f.write(f"{'='*60}\n\n")
                f.write(content)
                f.write(f"\n\n")
        
        self.logger.info(f"Saved plain text to: {output_file}")
        return str(output_file)
    
    def process_pdf(self, pdf_path: str, output_name: str = None) -> Dict:
        """Complete PDF processing pipeline"""
        if output_name is None:
            output_name = Path(pdf_path).stem
        
        self.logger.info(f"Starting PDF processing: {pdf_path}")
        
        # Extract text
        raw_text = self.extract_pdf_text(pdf_path)
        
        if not raw_text.strip():
            raise Exception("No text could be extracted from PDF")
        
        # Structure text
        sections = self.detect_sections(raw_text)
        
        if not sections:
            raise Exception("No sections could be created")
        
        # Save in multiple formats
        json_file = self.save_structured_text(sections, output_name)
        text_file = self.save_plain_text(sections, output_name)
        
        # Create summary
        total_words = sum(len(content.split()) for content in sections.values())
        
        summary = {
            "source_pdf": pdf_path,
            "output_name": output_name,
            "total_sections": len(sections),
            "total_words": total_words,
            "total_characters": len(raw_text),
            "structured_json": json_file,
            "plain_text": text_file,
            "sections": list(sections.keys())
        }
        
        summary_file = self.output_dir / f"{output_name}_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info("PDF PROCESSING COMPLETE")
        self.logger.info(f"Total sections: {len(sections)}")
        self.logger.info(f"Total words: {total_words:,}")
        self.logger.info(f"Output files saved in: {self.output_dir}")
        self.logger.info(f"{'='*60}")
        
        return summary

def main():
    """Main function for PDF text extraction"""
    # Configuration
    pdf_path = r"d:\Semester 7\Gen AI\Ass1\data\AI-Short Intro Book.pdf"
    output_name = "AI_Short_Introduction"
    
    # Create extractor
    extractor = PDFTextExtractor("extracted_text")
    
    try:
        # Process PDF
        summary = extractor.process_pdf(pdf_path, output_name)
        
        print("\n✅ PDF text extraction completed successfully!")
        print(f"📁 Output directory: {extractor.output_dir}")
        print(f"📖 Chapters extracted: {summary['total_chapters']}")
        print(f"📝 Total words: {summary['total_words']:,}")
        
        print("\n📄 Files created:")
        print(f"  - {Path(summary['structured_json']).name}")
        print(f"  - {Path(summary['plain_text']).name}")
        print(f"  - {output_name}_summary.json")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
