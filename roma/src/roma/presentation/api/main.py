"""
ROMA FastAPI Application.

Provides REST API endpoints matching v1 functionality for backward compatibility.
Uses separate schema layer for clean architecture.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import asyncio

from .schemas import (
    ExecuteRequest,
    ExecuteResponse,
    StreamEvent,
    SystemInfo,
    ValidationResponse,
    ProfileInfo,
    HealthResponse,
    SimpleResponse,
    StatusResponse,
)

from src.roma.framework_entry import (
    SentientAgent,
    ProfiledSentientAgent,
    LightweightSentientAgent,
    quick_research,
    quick_analysis,
    list_available_profiles
)


# FastAPI app initialization
app = FastAPI(
    title="ROMA v2 API",
    version="2.0.0", 
    description="Research-Oriented Multi-Agent Architecture",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React/Vite dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Core execution endpoints
@app.post("/api/execute", response_model=ExecuteResponse, tags=["Execution"])
async def execute_task(request: ExecuteRequest):
    """Execute any task through ROMA's intelligent agent system"""
    try:
        agent = ProfiledSentientAgent.create_with_profile(request.profile)
        result = agent.execute(
            request.goal, 
            enable_hitl=request.enable_hitl,
            max_steps=request.max_steps,
            **request.options
        )
        return ExecuteResponse(**result)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Execution failed: {str(e)}"
        )


@app.post("/api/async-execute", response_model=ExecuteResponse, tags=["Execution"])
async def async_execute_task(request: ExecuteRequest):
    """High-performance async task execution"""
    try:
        agent = LightweightSentientAgent.create_with_profile(request.profile)
        result = await agent.execute(
            request.goal,
            max_steps=request.max_steps,
            save_state=request.options.get("save_state", False),
            **request.options
        )
        return ExecuteResponse(**result)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Async execution failed: {str(e)}"
        )


# Simple API endpoints (v1 compatibility)
@app.post("/api/simple/execute", response_model=ExecuteResponse, tags=["Simple API"])
async def simple_execute(request: ExecuteRequest):
    """Simple execution endpoint for compatibility"""
    return await execute_goal(request)


@app.post("/api/simple/research", response_model=SimpleResponse, tags=["Simple API"])
async def simple_research(request: ExecuteRequest):
    """Quick research endpoint"""
    try:
        result = quick_research(
            request.goal,
            enable_hitl=request.enable_hitl,
            profile_name=request.profile
        )
        return SimpleResponse(result=result, status="completed")
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Research failed: {str(e)}"
        )


@app.post("/api/simple/analysis", response_model=SimpleResponse, tags=["Simple API"])
async def simple_analysis(request: ExecuteRequest):
    """Quick analysis endpoint"""
    try:
        result = quick_analysis(
            request.goal,
            enable_hitl=request.enable_hitl
        )
        return SimpleResponse(result=result, status="completed")
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


# System information endpoints
@app.get("/api/system-info", response_model=SystemInfo, tags=["System"])
async def get_system_info():
    """Get system information"""
    try:
        agent = SentientAgent.create()
        info = agent.get_system_info()
        return SystemInfo(**info)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system info: {str(e)}"
        )


@app.get("/api/simple/status", response_model=StatusResponse, tags=["Simple API"])
async def simple_status():
    """Simple API status endpoint"""
    return StatusResponse(
        available_profiles=list_available_profiles()
    )


# Configuration and validation
@app.get("/api/validate", response_model=ValidationResponse, tags=["Configuration"])
async def validate_configuration():
    """Validate system configuration"""
    try:
        agent = SentientAgent.create()
        validation = agent.validate_configuration()
        return ValidationResponse(**validation)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation failed: {str(e)}"
        )


# Profile management
@app.get("/api/profiles", response_model=List[str], tags=["Profiles"])
async def get_profiles():
    """List available profiles"""
    try:
        return list_available_profiles()
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list profiles: {str(e)}"
        )


@app.get("/api/profiles/{profile_name}", response_model=ProfileInfo, tags=["Profiles"])
async def get_profile_info(profile_name: str):
    """Get information about specific profile"""
    try:
        agent = ProfiledSentientAgent.create_with_profile(profile_name)
        info = agent.get_profile_info()
        return ProfileInfo(**info)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile '{profile_name}' not found: {str(e)}"
        )


# Health check
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse()


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "ROMA v2 API",
        "version": "2.0.0",
        "description": "Research-Oriented Multi-Agent Architecture",
        "documentation": "/docs",
        "status": "scaffolding",
        "endpoints": {
            "execute": "/api/execute",
            "research": "/api/simple/research", 
            "analysis": "/api/simple/analysis",
            "system_info": "/api/system-info",
            "profiles": "/api/profiles"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=True,
        log_level="info"
    )