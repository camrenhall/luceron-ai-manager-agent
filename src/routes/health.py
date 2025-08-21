"""
Basic health check routes for the Luceron AI Manager Agent.
"""
from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "operational",
        "service": "manager-agent",
        "version": "1.0.0"
    }


@router.get("/status")
async def status_check():
    """Simple status endpoint for load balancer."""
    return {"status": "running", "service": "manager-agent"}