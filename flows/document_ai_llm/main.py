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
from openai import OpenAI


@dataclass
class DocumentBlock:
    """Represents a text block from the document with its layout information"""

    text: str
    confidence: float
    page_number: int
    block_type: str  # paragraph, table, list, etc.
    bounding_box: Optional[Dict] = None


@dataclass
class InvoiceData:
    """Structured invoice data extracted by LLM"""
    
    invoice_number: Optional[str] = None
    issue_date: Optional[str] = None
    due_date: Optional[str] = None
    seller_name: Optional[str] = None
    seller_address: Optional[str] = None
    seller_tax_id: Optional[str] = None
    buyer_name: Optional[str] = None
    buyer_address: Optional[str] = None
    buyer_tax_id: Optional[str] = None
    total_amount: Optional[str] = None
    net_amount: Optional[str] = None
    tax_amount: Optional[str] = None
    currency: Optional[str] = None
    line_items: Optional[List[Dict]] = None
    payment_terms: Optional[str] = None
    payment_method: Optional[str] = None


class InvoiceAnalyzer:
    """OpenAI-powered invoice data extraction"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize the invoice analyzer
        
        Args:
            api_key: OpenAI API key (will use OPENAI_API_KEY env var if not provided)
            model: OpenAI model to use for analysis
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def extract_invoice_data(self, layout_text: str) -> InvoiceData:
        """
        Extract structured invoice data from layout-preserved text using OpenAI
        
        Args:
            layout_text: The layout-preserved text from document AI
            
        Returns:
            InvoiceData object with extracted information
        """
        prompt = self._create_extraction_prompt(layout_text)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at extracting structured data from invoices. Always respond with valid JSON format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return self._parse_extracted_data(result)
            
        except Exception as e:
            print(f"Error extracting invoice data with OpenAI: {e}")
            return InvoiceData()
    
    def _create_extraction_prompt(self, layout_text: str) -> str:
        """Create the prompt for invoice data extraction"""
        return f"""
Extract the most important information from this invoice text and return it as JSON.

Invoice text:
{layout_text}

Please extract and return ONLY a JSON object with the following fields (use null for missing information):
{{
    "invoice_number": "invoice/bill number",
    "issue_date": "date when invoice was issued (YYYY-MM-DD format if possible)",
    "due_date": "payment due date (YYYY-MM-DD format if possible)",
    "seller_name": "company/person issuing the invoice",
    "seller_address": "seller's full address",
    "seller_tax_id": "seller's tax ID/VAT number",
    "buyer_name": "company/person receiving the invoice", 
    "buyer_address": "buyer's full address",
    "buyer_tax_id": "buyer's tax ID/VAT number",
    "total_amount": "total amount to pay",
    "net_amount": "net amount before tax",
    "tax_amount": "tax amount",
    "currency": "currency code or symbol",
    "line_items": [
        {{
            "description": "item description",
            "quantity": "quantity",
            "unit_price": "price per unit",
            "total_price": "total price for this item",
            "tax_rate": "tax rate for this item"
        }}
    ],
    "payment_terms": "payment terms/conditions",
    "payment_method": "preferred payment method"
}}

Focus on accuracy and extract only information that is clearly present in the text.
"""
    
    def _parse_extracted_data(self, result: Dict) -> InvoiceData:
        """Parse the extracted JSON into InvoiceData object"""
        return InvoiceData(
            invoice_number=result.get("invoice_number"),
            issue_date=result.get("issue_date"),
            due_date=result.get("due_date"),
            seller_name=result.get("seller_name"),
            seller_address=result.get("seller_address"),
            seller_tax_id=result.get("seller_tax_id"),
            buyer_name=result.get("buyer_name"),
            buyer_address=result.get("buyer_address"),
            buyer_tax_id=result.get("buyer_tax_id"),
            total_amount=result.get("total_amount"),
            net_amount=result.get("net_amount"),
            tax_amount=result.get("tax_amount"),
            currency=result.get("currency"),
            line_items=result.get("line_items", []),
            payment_terms=result.get("payment_terms"),
            payment_method=result.get("payment_method")
        )


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
            type_="LAYOUT_PARSER_PROCESSOR",
        )

        try:
            response = self.client.create_processor(parent=parent, processor=processor)
            processor_id = response.name.split("/")[-1]
            print(f"Created processor with ID: {processor_id}")
            return processor_id
        except Exception as e:
            print(f"Error creating processor: {e}")
            raise

    def check_processor_status(self) -> Dict:
        """
        Check if the processor exists and get its details

        Returns:
            Dictionary with processor information
        """
        if not self.processor_id:
            return {"status": "error", "message": "No processor_id provided"}

        try:
            name = f"projects/{self.project_id}/locations/{self.location}/processors/{self.processor_id}"
            processor = self.client.get_processor(name=name)

            return {
                "status": "success",
                "processor_name": processor.display_name,
                "processor_type": processor.type_,
                "processor_state": processor.state.name
                if processor.state
                else "unknown",
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def list_processor_versions(self) -> Dict:
        """List available processor versions"""
        if not self.processor_id:
            return {"status": "error", "message": "No processor_id provided"}
            
        try:
            parent = f"projects/{self.project_id}/locations/{self.location}/processors/{self.processor_id}"
            request = documentai.ListProcessorVersionsRequest(parent=parent)
            response = self.client.list_processor_versions(request=request)
            
            versions = []
            for version in response.processor_versions:
                versions.append({
                    "name": version.name,
                    "display_name": version.display_name,
                    "state": version.state.name if version.state else "unknown"
                })
            
            return {"status": "success", "versions": versions}
        except Exception as e:
            return {"status": "error", "message": str(e)}

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

        # Prepare the request - try default version
        name = f"projects/{self.project_id}/locations/{self.location}/processors/{self.processor_id}"
        name = f"{name}/processorVersions/pretrained"
        print(f"DEBUG: Using processor: {name}")

        # Simplified request structure
        request = documentai.ProcessRequest(
            name=name,
            raw_document=documentai.RawDocument(content=content, mime_type=mime_type),
        )

        # Check processor status first
        processor_status = self.check_processor_status()
        print(f"Processor status: {processor_status}")
        
        # List available versions
        versions = self.list_processor_versions()
        print(f"Available versions: {versions}")

        # Process the document
        print(f"Processing document: {file_path.name}")
        print(f"Using processor: {name}")
        print(f"File size: {len(content)} bytes")
        print(f"MIME type: {mime_type}")

        try:
            result = self.client.process_document(request=request)
            print(f"DEBUG: API call successful")
            print(f"DEBUG: Result type: {type(result)}")
            print(f"DEBUG: Result attributes: {dir(result)}")
        except Exception as e:
            print(f"ERROR: Document AI API call failed: {e}")
            raise

        # Debug: Print document stats
        document = result.document
        print(f"DEBUG: Document type: {type(document)}")
        print(f"DEBUG: Document attributes: {[attr for attr in dir(document) if not attr.startswith('_')]}")
        print(f"Document text length: {len(document.text) if document.text else 0}")
        print(f"Number of pages: {len(document.pages)}")
        
        # Check for errors in the document
        if hasattr(document, 'error') and document.error:
            print(f"DEBUG: Document processing error: {document.error}")
        
        # Check response status
        if hasattr(result, 'human_review_status'):
            print(f"DEBUG: Human review status: {result.human_review_status}")

        # Debug: Print raw document structure
        if hasattr(document, "entities"):
            print(f"Number of entities: {len(document.entities)}")
        if hasattr(document, "text_styles"):
            print(
                f"Number of text styles: {len(document.text_styles) if document.text_styles else 0}"
            )

        # Check document_layout field for Layout Parser
        if hasattr(document, 'document_layout') and document.document_layout:
            print(f"DEBUG: Found document_layout field!")
            layout = document.document_layout
            print(f"DEBUG: Layout type: {type(layout)}")
            print(f"DEBUG: Layout attributes: {[attr for attr in dir(layout) if not attr.startswith('_')]}")
            
            if hasattr(layout, 'blocks'):
                print(f"DEBUG: Layout blocks: {len(layout.blocks)}")
                
        for i, page in enumerate(document.pages):
            print(
                f"Page {i+1}: {len(page.paragraphs)} paragraphs, {len(page.tables)} tables"
            )
            print(
                f"Page {i+1} dimension: {page.dimension.width if page.dimension else 'None'} x {page.dimension.height if page.dimension else 'None'}"
            )

            # Debug: Check if page has any content at all
            if hasattr(page, "blocks"):
                print(f"Page {i+1} blocks: {len(page.blocks)}")
            if hasattr(page, "tokens"):
                print(f"Page {i+1} tokens: {len(page.tokens)}")
            if hasattr(page, "lines"):
                print(f"Page {i+1} lines: {len(page.lines)}")

            # For Layout Parser, try checking visual elements
            if hasattr(page, "visual_elements"):
                print(f"Page {i+1} visual_elements: {len(page.visual_elements)}")

        # If document.text is empty but we have pages, try to construct text from page elements
        if not document.text and document.pages:
            print(
                f"DEBUG: document.text is empty, trying alternative extraction methods"
            )
            # This might be the issue - Layout Parser might not populate document.text
            # We need to extract text from page elements directly

        return document

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
        print(f"DEBUG: Starting text extraction")
        print(
            f"DEBUG: Document.text = '{document.text[:100] if document.text else 'None'}{'...' if document.text and len(document.text) > 100 else ''}'"
        )

        extracted_data = {
            "full_text": document.text or "",
            "confidence": 0.0,
            "pages": [],
            "blocks": [],
            "tables": [],
            "layout_preserved_text": "",
        }

        # Handle Layout Parser document_layout structure first
        if hasattr(document, 'document_layout') and document.document_layout and document.document_layout.blocks:
            print(f"DEBUG: Using Layout Parser document_layout with {len(document.document_layout.blocks)} blocks")
            full_text_parts = []
            
            for i, block in enumerate(document.document_layout.blocks):
                print(f"DEBUG: Processing layout block {i+1}")
                print(f"DEBUG: Block attributes: {[attr for attr in dir(block) if not attr.startswith('_')]}")
                
                # Extract text from layout block - Layout Parser might store text differently
                block_text = self._extract_text_from_layout_block(block)
                print(f"DEBUG: Block {i+1} text: '{block_text[:50] if block_text else 'Empty'}{'...' if block_text and len(block_text) > 50 else ''}'")
                
                if block_text:
                    full_text_parts.append(block_text)
                    
                    # Create block data
                    block_data = DocumentBlock(
                        text=block_text,
                        confidence=getattr(block, 'confidence', 0.0),
                        page_number=1,  # Layout parser doesn't have page concept
                        block_type=self._get_layout_block_type(block),
                        bounding_box=self._get_layout_block_bounding_box(block),
                    )
                    extracted_data["blocks"].append(asdict(block_data))
            
            # Combine all text
            if full_text_parts:
                extracted_data["full_text"] = "\n".join(full_text_parts)
                print(f"DEBUG: Extracted full text length: {len(extracted_data['full_text'])}")
                
                # Calculate average confidence
                if extracted_data["blocks"]:
                    extracted_data["confidence"] = sum(
                        b["confidence"] for b in extracted_data["blocks"]
                    ) / len(extracted_data["blocks"])
                
                # Generate layout-preserved text
                if preserve_layout:
                    extracted_data["layout_preserved_text"] = self._preserve_layout_formatting_from_blocks(extracted_data["blocks"])
                
                return extracted_data
            
        # Fall back to regular page processing if no document_layout
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
            print(
                f"DEBUG: Processing {len(page.paragraphs)} paragraphs on page {page_num + 1}"
            )
            for para_idx, paragraph in enumerate(page.paragraphs):
                print(f"DEBUG: Paragraph {para_idx + 1} layout: {paragraph.layout}")
                text = self._get_text_from_layout(paragraph.layout, document.text)
                print(
                    f"DEBUG: Extracted text from paragraph {para_idx + 1}: '{text[:50] if text else 'Empty'}{'...' if text and len(text) > 50 else ''}'"
                )
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
        print(
            f"DEBUG: _get_text_from_layout - layout.text_anchor: {layout.text_anchor}"
        )

        if layout.text_anchor and layout.text_anchor.text_segments:
            print(f"DEBUG: Found {len(layout.text_anchor.text_segments)} text segments")
            for i, segment in enumerate(layout.text_anchor.text_segments):
                start = int(segment.start_index) if segment.start_index else 0
                end = int(segment.end_index) if segment.end_index else len(full_text)
                segment_text = full_text[start:end]
                print(
                    f"DEBUG: Segment {i}: [{start}:{end}] = '{segment_text[:30] if segment_text else 'Empty'}{'...' if segment_text and len(segment_text) > 30 else ''}'"
                )
                text += segment_text
        else:
            print(f"DEBUG: No text_anchor or text_segments found")

        result = text.strip()
        print(
            f"DEBUG: Final extracted text: '{result[:30] if result else 'Empty'}{'...' if result and len(result) > 30 else ''}'"
        )
        return result

    def _extract_text_from_layout_block(self, block) -> str:
        """Extract text from a Layout Parser document layout block"""
        print(f"DEBUG: Layout block type: {type(block)}")
        
        # Check what attributes are available
        if hasattr(block, 'text_block') and block.text_block:
            print(f"DEBUG: Found text_block")
            text_block = block.text_block
            print(f"DEBUG: text_block attributes: {[attr for attr in dir(text_block) if not attr.startswith('_')]}")
            
            if hasattr(text_block, 'text'):
                return text_block.text or ""
            elif hasattr(text_block, 'blocks'):
                # Recursive text extraction
                parts = []
                for sub_block in text_block.blocks:
                    sub_text = self._extract_text_from_layout_block(sub_block)
                    if sub_text:
                        parts.append(sub_text)
                return " ".join(parts)
        
        # Handle table_block
        if hasattr(block, 'table_block') and block.table_block:
            print(f"DEBUG: Found table_block")
            table_block = block.table_block
            print(f"DEBUG: table_block attributes: {[attr for attr in dir(table_block) if not attr.startswith('_')]}")
            
            return self._extract_text_from_table_block(table_block)
        
        # Check for other possible text fields
        if hasattr(block, 'text') and block.text:
            return block.text
            
        return ""

    def _extract_text_from_table_block(self, table_block) -> str:
        """Extract text from a table block, formatting as structured table"""
        parts = []
        
        # Check if table_block has header_rows
        if hasattr(table_block, 'header_rows') and table_block.header_rows:
            print(f"DEBUG: Processing {len(table_block.header_rows)} header rows")
            for row_idx, row in enumerate(table_block.header_rows):
                row_text = self._extract_text_from_table_row(row)
                if row_text:
                    parts.append(f"HEADER ROW {row_idx + 1}: {row_text}")
        
        # Check if table_block has body_rows
        if hasattr(table_block, 'body_rows') and table_block.body_rows:
            print(f"DEBUG: Processing {len(table_block.body_rows)} body rows")
            for row_idx, row in enumerate(table_block.body_rows):
                row_text = self._extract_text_from_table_row(row)
                if row_text:
                    parts.append(f"ROW {row_idx + 1}: {row_text}")
        
        # If no structured rows, try to extract from cells directly
        if not parts and hasattr(table_block, 'cells') and table_block.cells:
            print(f"DEBUG: Processing {len(table_block.cells)} table cells directly")
            cell_texts = []
            for cell in table_block.cells:
                cell_text = self._extract_text_from_table_cell(cell)
                if cell_text:
                    cell_texts.append(cell_text)
            if cell_texts:
                parts.append(" | ".join(cell_texts))
        
        return "\n".join(parts) if parts else ""

    def _extract_text_from_table_row(self, row) -> str:
        """Extract text from a table row"""
        if hasattr(row, 'cells') and row.cells:
            cell_texts = []
            for cell in row.cells:
                cell_text = self._extract_text_from_table_cell(cell)
                if cell_text:
                    cell_texts.append(cell_text)
            return " | ".join(cell_texts)
        return ""

    def _extract_text_from_table_cell(self, cell) -> str:
        """Extract text from a table cell"""
        # Check if cell has a layout with text
        if hasattr(cell, 'layout') and cell.layout:
            if hasattr(cell.layout, 'text_anchor') and cell.layout.text_anchor:
                # This would need the full document text to resolve
                return "[CELL_TEXT]"  # Placeholder
        
        # Check if cell has direct text
        if hasattr(cell, 'text') and cell.text:
            return cell.text
            
        # Check if cell has blocks to process recursively
        if hasattr(cell, 'blocks') and cell.blocks:
            parts = []
            for block in cell.blocks:
                block_text = self._extract_text_from_layout_block(block)
                if block_text:
                    parts.append(block_text)
            return " ".join(parts)
        
        return ""

    def _get_layout_block_type(self, block) -> str:
        """Get the type of a Layout Parser block"""
        if hasattr(block, 'text_block') and block.text_block:
            if hasattr(block.text_block, 'type') and block.text_block.type:
                return str(block.text_block.type)
            return "text_block"
        elif hasattr(block, 'table_block') and block.table_block:
            return "table_block"
        elif hasattr(block, 'list_block') and block.list_block:
            return "list_block"
        return "layout_block"

    def _get_layout_block_bounding_box(self, block) -> Dict:
        """Get bounding box from Layout Parser block"""
        if hasattr(block, 'layout') and block.layout:
            return self._get_bounding_box(block.layout)
        return {}

    def _preserve_layout_formatting_from_blocks(self, blocks: List[Dict]) -> str:
        """Create layout-preserved text from blocks"""
        formatted_text = []
        formatted_text.append(f"\n{'='*50}")
        formatted_text.append(f"DOCUMENT LAYOUT")
        formatted_text.append(f"{'='*50}\n")
        
        for block in blocks:
            formatted_text.append(block["text"])
            formatted_text.append("")  # Empty line between blocks
            
        return "\n".join(formatted_text)

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

    def process_invoice(self, file_path: Union[str, Path], use_openai: bool = True, openai_api_key: Optional[str] = None) -> Dict:
        """
        Main method to process an invoice file

        Args:
            file_path: Path to the invoice file
            use_openai: Whether to use OpenAI for structured data extraction
            openai_api_key: OpenAI API key (optional, will use env var if not provided)

        Returns:
            Dictionary with extracted text, layout information, and structured data
        """
        try:
            # Process the document
            document = self.process_document(file_path)

            # Extract layout-preserved text
            extracted_data = self.extract_layout_text(document, preserve_layout=True)

            # Add metadata
            extracted_data["source_file"] = str(file_path)
            extracted_data["processor_id"] = self.processor_id

            # Use OpenAI to extract structured invoice data
            if use_openai and extracted_data.get("layout_preserved_text"):
                print("Extracting structured data using OpenAI...")
                try:
                    analyzer = InvoiceAnalyzer(api_key=openai_api_key)
                    invoice_data = analyzer.extract_invoice_data(extracted_data["layout_preserved_text"])
                    extracted_data["structured_data"] = asdict(invoice_data)
                    print("OpenAI extraction completed successfully")
                except Exception as e:
                    print(f"Warning: OpenAI extraction failed: {e}")
                    extracted_data["structured_data"] = None

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
        description="Extract text from invoices using Google Document AI Layout Parser and OpenAI"
    )
    parser.add_argument(
        "input_path", help="Path to invoice file or directory containing invoices"
    )
    parser.add_argument(
        "--project-id",
        help="Google Cloud Project ID (required)",
        required=True,
    )
    parser.add_argument(
        "--processor-id",
        help="Document AI Processor ID (create one if not provided)",
    )
    parser.add_argument(
        "--location", default="eu", choices=["us", "eu"], help="Processor location"
    )
    parser.add_argument(
        "--output-dir",
        default="invoices/document_ai_llm/results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--openai-api-key",
        help="OpenAI API key (can also be set via OPENAI_API_KEY environment variable)",
    )
    parser.add_argument(
        "--no-openai",
        action="store_true",
        help="Skip OpenAI structured data extraction",
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
        results = parser.process_invoice(
            input_path, 
            use_openai=not args.no_openai,
            openai_api_key=args.openai_api_key
        )

        output_file = output_dir / f"{input_path.stem}_extracted.json"
        save_results(results, output_file)

        # Also save the layout-preserved text separately for easy viewing
        text_file = output_dir / f"{input_path.stem}_layout.txt"
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(results["layout_preserved_text"])
        print(f"Layout text saved to: {text_file}")
        
        # Save structured data separately if available
        if results.get("structured_data"):
            structured_file = output_dir / f"{input_path.stem}_structured.json"
            with open(structured_file, "w", encoding="utf-8") as f:
                json.dump(results["structured_data"], f, ensure_ascii=False, indent=2)
            print(f"Structured data saved to: {structured_file}")

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
                results = parser.process_invoice(
                    invoice_file,
                    use_openai=not args.no_openai,
                    openai_api_key=args.openai_api_key
                )

                output_file = output_dir / f"{invoice_file.stem}_extracted.json"
                save_results(results, output_file)

                text_file = output_dir / f"{invoice_file.stem}_layout.txt"
                with open(text_file, "w", encoding="utf-8") as f:
                    f.write(results["layout_preserved_text"])
                
                # Save structured data separately if available
                if results.get("structured_data"):
                    structured_file = output_dir / f"{invoice_file.stem}_structured.json"
                    with open(structured_file, "w", encoding="utf-8") as f:
                        json.dump(results["structured_data"], f, ensure_ascii=False, indent=2)

            except Exception as e:
                print(f"Error processing {invoice_file.name}: {e}")

    else:
        print(f"Error: {input_path} is neither a file nor a directory")
        return 1

    print("\nProcessing complete!")
    return 0


if __name__ == "__main__":
    exit(main())
