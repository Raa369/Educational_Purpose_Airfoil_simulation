import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from solver import airfoil_coords, normalize, simulate

app = FastAPI(title="Airfoil CFD 2D")
slots = threading.Semaphore(2)  # batasi simulasi paralel agar server hosting tidak kewalahan


class Params(BaseModel):
    m: float = Field(0.02, ge=0, le=0.09)    # camber maks (fraksi chord)
    p: float = Field(0.4, ge=0.1, le=0.9)    # posisi camber maks
    t: float = Field(0.12, ge=0.06, le=0.25)  # tebal maks
    alpha: float = Field(4, ge=-10, le=15)   # sudut serang (derajat)
    Re: float = Field(1000, ge=200, le=3000)
    steps: int = Field(800, ge=300, le=1500)
    coords: list[list[float]] | None = Field(None, max_length=2000)  # airfoil custom (hasil import .dat)


@app.post("/api/airfoil")
def airfoil(q: Params):
    try:
        x, y = normalize(q.coords) if q.coords else airfoil_coords(q.m, q.p, q.t)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"x": x.tolist(), "y": y.tolist()}


@app.post("/api/simulate")
def run(q: Params):
    if not slots.acquire(blocking=False):
        raise HTTPException(429, "Server sedang sibuk, coba lagi sebentar.")
    try:
        return simulate(q.m, q.p, q.t, q.alpha, q.Re, q.steps, coords=q.coords)
    except ValueError as e:
        raise HTTPException(422, str(e))
    finally:
        slots.release()


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")