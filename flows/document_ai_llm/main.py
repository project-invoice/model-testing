#!/usr/bin/env python3
"""
Google Document AI Layout Parser for Invoice OCR
This script extracts text from invoice images using Google Document AI's Layout Parser,
preserving the document structure for better downstream processing with LLMs.
"""

import os
import json
import base64
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict
import argparse

from google.cloud import documentai_v1 as documentai
from google.api_core.client_options import ClientOptions


@dataclass
class DocumentBlock:
    """Represents a text block from the document with its layout information"""

    text: str
    confidence: float
    page_number: int
    block_type: str  # paragraph, table, list, etc.
    bounding_box: Optional[Dict] = None


class DocumentAILayoutParser:
    """
    Wrapper for Google Document AI Layout Parser to extract structured text from documents
    """

    def __init__(
        self,
        project_id: str,
        location: str = "us",  # or "eu" for European region
        processor_id: Optional[str] = None,
        processor_version: Optional[str] = None,
    ):
        """
        Initialize the Document AI Layout Parser

        Args:
            project_id: Google Cloud Project ID
            location: Processor location (us or eu)
            processor_id: Document AI processor ID (if you have one created)
            processor_version: Optional specific version of the processor
        """
        self.project_id = project_id
        self.location = location
        self.processor_id = processor_id
        self.processor_version = processor_version or "rc"

        # Initialize the Document AI client
        opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
        self.client = documentai.DocumentProcessorServiceClient(client_options=opts)

        # If no processor_id provided, you'll need to create one or use existing
        if not self.processor_id:
            print(
                "Warning: No processor_id provided. Please create a Layout Parser processor in GCP Console."
            )
            print("Visit: https://console.cloud.google.com/ai/document-ai/processors")

    def create_processor(self, display_name: str = "Invoice Layout Parser") -> str:
        """
        Create a new Layout Parser processor (run this once to set up)

        Args:
            display_name: Name for the processor in GCP Console

        Returns:
            Processor ID
        """
        parent = f"projects/{self.project_id}/locations/{self.location}"

        processor = documentai.Processor(
            display_name=display_name,
            type_="LAYOUT_PARSER_PROCESSOR",  # This is the type for layout parsing
        )

        try:
            response = self.client.create_processor(parent=parent, processor=processor)
            processor_id = response.name.split("/")[-1]
            print(f"Created processor with ID: {processor_id}")
            return processor_id
        except Exception as e:
            print(f"Error creating processor: {e}")
            raise

    def process_document(
        self, file_path: Union[str, Path], mime_type: Optional[str] = None
    ) -> documentai.Document:
        """
        Process a document file using Document AI Layout Parser

        Args:
            file_path: Path to the document file
            mime_type: MIME type of the document (auto-detected if not provided)

        Returns:
            Processed Document object
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Auto-detect MIME type if not provided
        if not mime_type:
            mime_type = self._get_mime_type(file_path)

        # Read the file
        with open(file_path, "rb") as f:
            content = f.read()

        # Prepare the request
        name = f"projects/{self.project_id}/locations/{self.location}/processors/{self.processor_id}"
        if self.processor_version != "rc":
            name = f"{name}/processorVersions/{self.processor_version}"

        document = documentai.Document(content=content, mime_type=mime_type)

        request = documentai.ProcessRequest(
            name=name,
            raw_document=documentai.RawDocument(content=content, mime_type=mime_type),
        )

        # Process the document
        print(f"Processing document: {file_path.name}")
        result = self.client.process_document(request=request)

        return result.document

    def extract_layout_text(
        self, document: documentai.Document, preserve_layout: bool = True
    ) -> Dict:
        """
        Extract text from the document preserving layout structure

        Args:
            document: Processed Document object from Document AI
            preserve_layout: Whether to preserve spatial layout in text

        Returns:
            Dictionary containing extracted text and metadata
        """
        extracted_data = {
            "full_text": document.text,
            "confidence": 0.0,
            "pages": [],
            "blocks": [],
            "tables": [],
            "layout_preserved_text": "",
        }

        # Process each page
        for page_num, page in enumerate(document.pages):
            page_data = {
                "page_number": page_num + 1,
                "width": page.dimension.width if page.dimension else None,
                "height": page.dimension.height if page.dimension else None,
                "blocks": [],
                "tables": [],
            }

            # Extract paragraphs
            for paragraph in page.paragraphs:
                text = self._get_text_from_layout(paragraph.layout, document.text)
                if text:
                    block = DocumentBlock(
                        text=text,
                        confidence=paragraph.layout.confidence,
                        page_number=page_num + 1,
                        block_type="paragraph",
                        bounding_box=self._get_bounding_box(paragraph.layout),
                    )
                    page_data["blocks"].append(asdict(block))
                    extracted_data["blocks"].append(asdict(block))

            # Extract tables
            for table in page.tables:
                table_data = self._extract_table(table, document.text, page_num + 1)
                if table_data:
                    page_data["tables"].append(table_data)
                    extracted_data["tables"].append(table_data)

            extracted_data["pages"].append(page_data)

        # Calculate average confidence
        if extracted_data["blocks"]:
            extracted_data["confidence"] = sum(
                b["confidence"] for b in extracted_data["blocks"]
            ) / len(extracted_data["blocks"])

        # Generate layout-preserved text
        if preserve_layout:
            extracted_data["layout_preserved_text"] = self._preserve_layout_formatting(
                extracted_data["pages"]
            )

        return extracted_data

    def _get_text_from_layout(
        self, layout: documentai.Document.Page.Layout, full_text: str
    ) -> str:
        """Extract text from a layout element using text segments"""
        text = ""
        if layout.text_anchor and layout.text_anchor.text_segments:
            for segment in layout.text_anchor.text_segments:
                start = int(segment.start_index) if segment.start_index else 0
                end = int(segment.end_index) if segment.end_index else len(full_text)
                text += full_text[start:end]
        return text.strip()

    def _get_bounding_box(self, layout: documentai.Document.Page.Layout) -> Dict:
        """Extract bounding box coordinates from layout"""
        if not layout.bounding_poly or not layout.bounding_poly.vertices:
            return {}

        vertices = layout.bounding_poly.vertices
        return {
            "x_min": min(v.x for v in vertices),
            "y_min": min(v.y for v in vertices),
            "x_max": max(v.x for v in vertices),
            "y_max": max(v.y for v in vertices),
        }

    def _extract_table(
        self, table: documentai.Document.Page.Table, full_text: str, page_num: int
    ) -> Dict:
        """Extract table data with structure"""
        table_data = {
            "page_number": page_num,
            "rows": [],
            "confidence": table.layout.confidence if table.layout else 0.0,
        }

        # Group cells by row
        rows_dict = {}
        for cell in table.body_rows:
            for table_cell in cell.cells:
                row_idx = (
                    table_cell.layout.row_span.start
                    if table_cell.layout.row_span
                    else 0
                )
                if row_idx not in rows_dict:
                    rows_dict[row_idx] = []

                cell_text = self._get_text_from_layout(table_cell.layout, full_text)
                rows_dict[row_idx].append(cell_text)

        # Convert to list of rows
        for row_idx in sorted(rows_dict.keys()):
            table_data["rows"].append(rows_dict[row_idx])

        return table_data

    def _preserve_layout_formatting(self, pages: List[Dict]) -> str:
        """
        Create a text representation that preserves spatial layout
        This helps LLMs better understand document structure
        """
        formatted_text = []

        for page in pages:
            formatted_text.append(f"\n{'='*50}")
            formatted_text.append(f"PAGE {page['page_number']}")
            formatted_text.append(f"{'='*50}\n")

            # Add paragraphs
            for block in page.get("blocks", []):
                formatted_text.append(block["text"])
                formatted_text.append("")  # Empty line between blocks

            # Add tables
            for table in page.get("tables", []):
                formatted_text.append("\n[TABLE]")
                for row in table["rows"]:
                    formatted_text.append(" | ".join(str(cell) for cell in row))
                formatted_text.append("[/TABLE]\n")

        return "\n".join(formatted_text)

    def _get_mime_type(self, file_path: Path) -> str:
        """Auto-detect MIME type from file extension"""
        ext = file_path.suffix.lower()
        mime_types = {
            ".pdf": "application/pdf",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".tiff": "image/tiff",
            ".tif": "image/tiff",
            ".bmp": "image/bmp",
            ".gif": "image/gif",
        }
        return mime_types.get(ext, "application/pdf")

    def process_invoice(self, file_path: Union[str, Path]) -> Dict:
        """
        Main method to process an invoice file

        Args:
            file_path: Path to the invoice file

        Returns:
            Dictionary with extracted text and layout information
        """
        try:
            # Process the document
            document = self.process_document(file_path)

            # Extract layout-preserved text
            extracted_data = self.extract_layout_text(document, preserve_layout=True)

            # Add metadata
            extracted_data["source_file"] = str(file_path)
            extracted_data["processor_id"] = self.processor_id

            return extracted_data

        except Exception as e:
            print(f"Error processing invoice: {e}")
            raise


def save_results(results: Dict, output_path: Path):
    """Save extraction results to JSON file"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract text from invoices using Google Document AI Layout Parser"
    )
    parser.add_argument(
        "input_path", help="Path to invoice file or directory containing invoices"
    )
    parser.add_argument(
        "--project-id",
        help="Google Cloud Project ID",
        default="easy-invoice-469917",
    )
    parser.add_argument(
        "--processor-id",
        help="Document AI Processor ID (create one if not provided)",
        default="14e9217d88542adb",
    )
    parser.add_argument(
        "--location", default="eu", choices=["us", "eu"], help="Processor location"
    )
    parser.add_argument(
        "--output-dir",
        default="invoices/document_ai_llm/results",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Initialize the parser
    parser = DocumentAILayoutParser(
        project_id=args.project_id,
        location=args.location,
        processor_id=args.processor_id,
    )

    # Process input
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)

    if input_path.is_file():
        # Process single file
        print(f"Processing invoice: {input_path}")
        results = parser.process_invoice(input_path)

        output_file = output_dir / f"{input_path.stem}_extracted.json"
        save_results(results, output_file)

        # Also save the layout-preserved text separately for easy viewing
        text_file = output_dir / f"{input_path.stem}_layout.txt"
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(results["layout_preserved_text"])
        print(f"Layout text saved to: {text_file}")

    elif input_path.is_dir():
        # Process all images in directory
        image_extensions = [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif"]
        invoice_files = [
            f for f in input_path.iterdir() if f.suffix.lower() in image_extensions
        ]

        print(f"Found {len(invoice_files)} invoice files to process")

        for invoice_file in invoice_files:
            print(f"\nProcessing: {invoice_file.name}")
            try:
                results = parser.process_invoice(invoice_file)

                output_file = output_dir / f"{invoice_file.stem}_extracted.json"
                save_results(results, output_file)

                text_file = output_dir / f"{invoice_file.stem}_layout.txt"
                with open(text_file, "w", encoding="utf-8") as f:
                    f.write(results["layout_preserved_text"])

            except Exception as e:
                print(f"Error processing {invoice_file.name}: {e}")

    else:
        print(f"Error: {input_path} is neither a file nor a directory")
        return 1

    print("\nProcessing complete!")
    return 0


if __name__ == "__main__":
    exit(main())
