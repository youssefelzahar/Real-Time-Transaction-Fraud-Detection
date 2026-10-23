"""NilePay fraud scoring API.

Request contract: the client sends the transaction itself (type, amount, sender and
receiver account IDs). Pre-transaction balances come from the AccountService, which is
the source of truth. `sender_balance` / `receiver_balance` in the request are only a
fallback for accounts the service does not know (e.g. local testing in Swagger); the
response reports which source each balance came from.

Run:  MODEL_DIR=models/e6_v1 uvicorn api.main:app --reload
"""
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal, Optional, Protocol

from fastapi import Body, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.predict import FraudPredictor
from time import perf_counter

class AccountService(Protocol):
    def get_balance(self, account_id: str) -> Optional[float]:
        """Current balance before the transaction, or None if unknown."""


class InMemoryAccountService:
    """Local stand-in for the core-banking / account service.
    Optionally seeded from a JSON file {account_id: balance} via ACCOUNT_SNAPSHOT."""

    def __init__(self, balances=None):
        self.balances = dict(balances or {})

    @classmethod
    def from_env(cls):
        path = os.getenv("ACCOUNT_SNAPSHOT")
        return cls(json.loads(Path(path).read_text())) if path else cls()

    def get_balance(self, account_id):
        return self.balances.get(account_id)


class TransactionRequest(BaseModel):
    transaction_id: Optional[str] = None
    type: Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    amount: float = Field(ge=0)
    sender_id: str = Field(min_length=1, description="nameOrig")
    receiver_id: str = Field(min_length=1, description="nameDest")
    step: Optional[int] = Field(default=None, ge=0, description="Hours since start; only used by E5")
    sender_balance: Optional[float] = Field(
        default=None, ge=0, description="oldbalanceOrg fallback, used only if the account service has no balance")
    receiver_balance: Optional[float] = Field(
        default=None, ge=0, description="oldbalanceDest fallback, used only if the account service has no balance")


class ScoreResponse(BaseModel):
    transaction_id: Optional[str]
    fraud_probability: float
    decision: Literal["approve", "review"]
    threshold: float
    model_version: str
    missing_features: list[str]
    balance_source: dict[str, Literal["account_service", "request", "missing"]]


EXAMPLES = {
    "drained_transfer": {
        "summary": "Suspicious: TRANSFER emptying the sender to an empty account",
        "value": {"transaction_id": "tx-001", "type": "TRANSFER", "amount": 181.0,
                  "sender_id": "C1305486145", "receiver_id": "C553264065",
                  "sender_balance": 181.0, "receiver_balance": 0.0},
    },
    "max_amount_cash_out": {
        "summary": "Suspicious: CASH_OUT at the 10M transaction cap",
        "value": {"transaction_id": "tx-002", "type": "CASH_OUT", "amount": 10000000.0,
                  "sender_id": "C1000000001", "receiver_id": "C2000000002",
                  "sender_balance": 10000000.0, "receiver_balance": 0.0},
    },
    "normal_payment": {
        "summary": "Normal: PAYMENT to a merchant",
        "value": {"transaction_id": "tx-003", "type": "PAYMENT", "amount": 9839.64,
                  "sender_id": "C1231006815", "receiver_id": "M1979787155",
                  "sender_balance": 170136.0, "receiver_balance": 0.0},
    },
    "normal_transfer": {
        "summary": "Normal: TRANSFER of part of a balance to a funded account",
        "value": {"transaction_id": "tx-004", "type": "TRANSFER", "amount": 25000.0,
                  "sender_id": "C1111111111", "receiver_id": "C2222222222",
                  "sender_balance": 300000.0, "receiver_balance": 150000.0},
    },
    "cash_in": {
        "summary": "Normal: CASH_IN deposit",
        "value": {"transaction_id": "tx-005", "type": "CASH_IN", "amount": 5000.0,
                  "sender_id": "C3333333333", "receiver_id": "C4444444444",
                  "sender_balance": 1200.0, "receiver_balance": 80000.0},
    },
    "unknown_balances": {
        "summary": "No balances anywhere: scored with missing features",
        "value": {"transaction_id": "tx-006", "type": "TRANSFER", "amount": 5000.0,
                  "sender_id": "C9999999999", "receiver_id": "C8888888888"},
    },
    "e5_with_step": {
        "summary": "For E5 models only: include step (hour index)",
        "value": {"transaction_id": "tx-007", "type": "TRANSFER", "amount": 181.0,
                  "sender_id": "C1305486145", "receiver_id": "C553264065", "step": 1,
                  "sender_balance": 181.0, "receiver_balance": 0.0},
    },
}


def resolve_balance(accounts, account_id, fallback):
    balance = accounts.get_balance(account_id)
    if balance is not None:
        return balance, "account_service"
    if fallback is not None:
        return fallback, "request"
    return None, "missing"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not hasattr(app.state, "predictor"):
        app.state.predictor = FraudPredictor.load(os.getenv("MODEL_DIR", "models/e6_v1"))
    if not hasattr(app.state, "account_service"):
        app.state.account_service = InMemoryAccountService.from_env()
    yield


app = FastAPI(title="NilePay Fraud Scoring", lifespan=lifespan)


@app.get("/health")
def health(request: Request):
    predictor = request.app.state.predictor
    return {"status": "ok", "model_version": predictor.model_version}



@app.post("/predict", response_model=ScoreResponse)
def predict(
    tx: Annotated[TransactionRequest, Body(openapi_examples=EXAMPLES)],
    request: Request,
):
    start = perf_counter()

    predictor = request.app.state.predictor
    accounts = request.app.state.account_service

    if "step" in predictor.pipeline.required_inputs and tx.step is None:
        raise HTTPException(
            status_code=422,
            detail="This model version requires 'step'",
        )

    balance_start = perf_counter()

    sender_balance, sender_source = resolve_balance(
        accounts, tx.sender_id, tx.sender_balance
    )
    receiver_balance, receiver_source = resolve_balance(
        accounts, tx.receiver_id, tx.receiver_balance
    )

    balance_ms = (perf_counter() - balance_start) * 1000

    record = {
        "type": tx.type,
        "amount": tx.amount,
        "oldbalanceOrg": sender_balance,
        "oldbalanceDest": receiver_balance,
        "step": tx.step,
    }

    model_start = perf_counter()
    result = predictor.score(record)
    model_ms = (perf_counter() - model_start) * 1000

    total_ms = (perf_counter() - start) * 1000

    print({
        "balance_ms": round(balance_ms, 3),
        "model_ms": round(model_ms, 3),
        "total_ms": round(total_ms, 3),
    })

    return ScoreResponse(
        transaction_id=tx.transaction_id,
        **result,
        balance_source={
            "sender": sender_source,
            "receiver": receiver_source,
        },
    )
