"""Visualization endpoints."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from roma_dspy.api.schemas import VisualizationRequest, VisualizationResponse
from roma_dspy.api.dependencies import get_storage, get_config_manager, verify_execution_exists
from roma_dspy.core.storage.postgres_storage import PostgresStorage
from roma_dspy.config.manager import ConfigManager
from roma_dspy.core.engine.dag import TaskDAG
from roma_dspy.types import ExecutionStatus
from roma_dspy.visualizer import (
    TreeVisualizer,
    TimelineVisualizer,
    StatisticsVisualizer,
    ContextFlowVisualizer,
    LLMTraceVisualizer,
)

router = APIRouter()


@router.post("/executions/{execution_id}/visualize", response_model=VisualizationResponse)
async def visualize_execution(
    viz_request: VisualizationRequest,
    execution_id: str = Depends(verify_execution_exists),
    storage: PostgresStorage = Depends(get_storage),
    config_manager: ConfigManager = Depends(get_config_manager)
) -> VisualizationResponse:
    """
    Generate visualization for an execution.

    Args:
        execution_id: Execution ID
        viz_request: Visualization options

    Returns:
        Visualization content in requested format
    """
    try:
        # Get execution
        execution = await storage.get_execution(execution_id)

        if not execution:
            raise HTTPException(
                status_code=404,
                detail=f"Execution {execution_id} not found"
            )

        # Get latest checkpoint for DAG snapshot
        checkpoint = await storage.get_latest_checkpoint(execution_id, valid_only=True)

        if not checkpoint:
            # Provide helpful error messages based on execution status
            if execution.status == ExecutionStatus.RUNNING.value:
                detail = (
                    f"Execution {execution_id} is still running and has not yet created a checkpoint. "
                    f"Wait for the execution to make progress before visualizing."
                )
            elif execution.status in (ExecutionStatus.FAILED.value, ExecutionStatus.CANCELLED.value):
                detail = (
                    f"Execution {execution_id} was {execution.status} before creating any checkpoints. "
                    f"The execution may have been interrupted early. Try running the task again."
                )
            else:
                detail = (
                    f"Execution {execution_id} has no valid checkpoints available (status: {execution.status}). "
                    f"This may indicate the execution was interrupted or never started properly."
                )

            raise HTTPException(
                status_code=400,
                detail=detail
            )

        # Convert DAGSnapshot to dict for visualization
        dag_snapshot = checkpoint.root_dag
        if hasattr(dag_snapshot, 'model_dump'):
            snapshot_dict = dag_snapshot.model_dump(mode="python")
        elif isinstance(dag_snapshot, dict):
            snapshot_dict = dag_snapshot
        else:
            raise ValueError(f"Unexpected dag_snapshot type: {type(dag_snapshot)}")

        # Get visualization options (use defaults if not provided)
        from roma_dspy.api.schemas import VisualizationOptions
        opts = viz_request.options or VisualizationOptions()

        # If verbose flag is set, enable all details
        if opts.verbose:
            opts.show_ids = True
            opts.show_timing = True
            opts.show_tokens = True
            if opts.max_goal_length == 60:  # Only override if still default
                opts.max_goal_length = 0  # Unlimited

        # Select visualizer
        visualizer_type = viz_request.visualizer_type.lower()

        if visualizer_type == "tree":
            visualizer = TreeVisualizer(
                show_ids=opts.show_ids,
                show_timing=opts.show_timing,
                show_tokens=opts.show_tokens,
                max_goal_length=opts.max_goal_length
            )
        elif visualizer_type == "timeline":
            visualizer = TimelineVisualizer()
        elif visualizer_type == "statistics":
            visualizer = StatisticsVisualizer()
        elif visualizer_type == "context_flow":
            visualizer = ContextFlowVisualizer()
        elif visualizer_type == "llm_trace":
            # Get MLflow tracking URI from config with defensive checks
            config = config_manager.load_config(profile=viz_request.profile)

            # Safely extract MLflow tracking URI with fallback to default
            mlflow_tracking_uri = "http://127.0.0.1:5000"  # Default
            if hasattr(config, 'observability') and config.observability is not None:
                if hasattr(config.observability, 'mlflow') and config.observability.mlflow is not None:
                    mlflow_tracking_uri = getattr(config.observability.mlflow, 'tracking_uri', mlflow_tracking_uri)

            # Pull experiment name from config profile
            exp_name = None
            if hasattr(config, 'observability') and config.observability is not None:
                if hasattr(config.observability, 'mlflow') and config.observability.mlflow is not None:
                    exp_name = getattr(config.observability.mlflow, 'experiment_name', None)

            visualizer = LLMTraceVisualizer(
                verbose=opts.verbose,
                fancy=opts.fancy,
                mlflow_tracking_uri=mlflow_tracking_uri,
                mlflow_experiment_name=exp_name,
                show_io=opts.show_io,
                console_width=getattr(opts, 'width', None),
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown visualizer type: {visualizer_type}. "
                       f"Valid options: tree, timeline, statistics, context_flow, llm_trace"
            )

        # Generate visualization based on data source
        if viz_request.format == "text":
            # Route based on data source
            if viz_request.data_source == "mlflow" and visualizer_type == "llm_trace":
                # Use MLflow traces for rich DSPy trace visualization
                try:
                    content = visualizer.visualize_from_mlflow(execution_id)
                except Exception as e:
                    logger.warning(f"MLflow visualization failed, falling back to checkpoint: {e}")
                    content = visualizer.visualize_from_snapshot(snapshot_dict)
            else:
                # Use checkpoint snapshot visualization (default)
                content = visualizer.visualize_from_snapshot(snapshot_dict)
        elif viz_request.format == "json":
            import json
            if viz_request.data_source == "mlflow" and visualizer_type == "llm_trace":
                try:
                    data = visualizer.build_mlflow_trace_data(execution_id)
                except Exception as e:
                    logger.warning(f"MLflow JSON visualization failed: {e}. Falling back to checkpoint data.")
                    data = {"warning": str(e), "snapshot": snapshot_dict}
                content = json.dumps(data, indent=2, default=str)
            else:
                # Return checkpoint snapshot structure by default
                content = json.dumps(snapshot_dict, indent=2, default=str)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown format: {viz_request.format}. Valid options: text, json"
            )

        return VisualizationResponse(
            execution_id=execution_id,
            visualizer_type=visualizer_type,
            content=content,
            format=viz_request.format,
            generated_at=datetime.now(timezone.utc)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to visualize execution {execution_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate visualization: {str(e)}"
        )


@router.get("/executions/{execution_id}/dag", response_model=dict)
async def get_dag_snapshot(
    execution_id: str = Depends(verify_execution_exists),
    storage: PostgresStorage = Depends(get_storage)
) -> dict:
    """
    Get raw DAG snapshot for an execution.

    Args:
        execution_id: Execution ID

    Returns:
        DAG snapshot as JSON
    """
    try:
        # Get execution
        execution = await storage.get_execution(execution_id)

        if not execution:
            raise HTTPException(
                status_code=404,
                detail=f"Execution {execution_id} not found"
            )

        # Get latest checkpoint for DAG snapshot
        checkpoint = await storage.get_latest_checkpoint(execution_id, valid_only=True)

        if not checkpoint:
            # Provide helpful error messages based on execution status
            if execution.status == ExecutionStatus.RUNNING.value:
                detail = (
                    f"Execution {execution_id} is still running and has not yet created a checkpoint. "
                    f"Wait for the execution to make progress before retrieving the DAG."
                )
            elif execution.status in (ExecutionStatus.FAILED.value, ExecutionStatus.CANCELLED.value):
                detail = (
                    f"Execution {execution_id} was {execution.status} before creating any checkpoints. "
                    f"The execution may have been interrupted early."
                )
            else:
                detail = (
                    f"Execution {execution_id} has no valid checkpoints available (status: {execution.status}). "
                    f"This may indicate the execution was interrupted or never started properly."
                )

            raise HTTPException(
                status_code=400,
                detail=detail
            )

        # Convert DAGSnapshot model to dict if needed
        dag_data = checkpoint.root_dag
        if hasattr(dag_data, 'model_dump'):
            return dag_data.model_dump(mode="python")
        return dag_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get DAG snapshot for {execution_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get DAG snapshot: {str(e)}"
        )
