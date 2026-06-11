from pydantic import BaseModel
import datetime
from typing import Optional, List, Union, Dict, Any

class Item(BaseModel):
    item_name: str
    item_code: Optional[str] = None
    quantity: int
    discount: float = 0.0
    price: float
    remarks: Optional[str] = None

    @property
    def subtotal(self):
        return self.quantity * self.price

class UploadRequest(BaseModel):
    customer_id: int
    batch_id: Optional[str] = None
    img_urls: List[str]

class DraftReceiptCreate(BaseModel):
    store_name: Optional[str] = None
    receipt_no: Optional[str] = None
    receipt_date: Optional[str] = None
    customer_id: int
    total: Optional[float] = None
    total_items: Optional[int] = None
    tax: Optional[float] = None
    commission_rate: Optional[float] = None
    commission_amount: Optional[float] = None
    remarks: Optional[str] = None
    items: Optional[List[Union[Item, dict]]] = None
    img_url: Optional[Union[str, List[str]]] = None
    status: Optional[str] = "pending_review"
    batch_id: Optional[str] = None
    raw_transcribe: Optional[str] = None
    raw_receipt: Optional[Dict[str, Any]] = None
