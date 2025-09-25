"""
ROMA CLI using Click with Hydra integration.

Provides command-line interface for ROMA v2 with clean commands and options.
Uses Hydra configuration with top-down propagation.
"""

import asyncio
from typing import Any

import click
from hydra import compose, initialize
from omegaconf import OmegaConf

from roma.application.orchestration.system_manager import SystemManager
from roma.domain.value_objects.config.roma_config import ROMAConfig
from roma.framework_entry import (
    LightweightSentientAgent,
    SentientAgent,
    list_available_profiles,
    quick_research,
)


def get_roma_config(_config_path: str | None = None, profile: str | None = None) -> ROMAConfig:
    """Get Hydra configuration and create validated ROMAConfig object."""
    try:
        # Initialize Hydra
        with initialize(version_base=None, config_path="../../../config"):
            # Compose config with overrides
            overrides = []
            if profile:
                overrides.append(f"profile={profile}")

            cfg = compose(config_name="config", overrides=overrides)

            # Convert to container dict and create ROMAConfig with validation
            config_dict = OmegaConf.to_container(cfg, resolve=True)
            if not isinstance(config_dict, dict):
                raise ValueError("Configuration must be a dictionary")
            return ROMAConfig.from_dict(config_dict)

    except Exception as e:
        click.echo(f"⚠️  Could not load Hydra config: {e}")
        # Return minimal test config
        test_config_dict = {
            "app": {"name": "roma", "version": "2.0.0", "environment": "dev"},
            "profile": {"name": profile or "test_profile"},
            "default_profile": profile or "test_profile",
        }
        return ROMAConfig.from_dict(test_config_dict)


@click.group()
@click.version_option(version="2.0.0", prog_name="ROMA")
@click.option("--config", "-c", help="Configuration file path")
@click.option("--profile", "-p", help="Profile to use")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx: click.Context, config: str | None, profile: str | None, verbose: bool) -> None:
    """ROMA v2 - Recursive Orchestration Multi-Agent Architecture"""
    # Create and store all objects for subcommands
    profile = profile or "test_profile"
    roma_config = get_roma_config(config, profile)
    system_manager = SystemManager(roma_config)

    # Store all objects in context
    ctx.ensure_object(dict)
    ctx.obj["system_manager"] = system_manager
    ctx.obj["config"] = roma_config
    ctx.obj["profile"] = profile
    ctx.obj["verbose"] = verbose

    if verbose:
        click.echo("🔧 Verbose mode enabled")
        click.echo(f"📋 Profile: {profile}")
        click.echo(
            f"⚙️  Config validated and loaded: {roma_config.app.name} v{roma_config.app.version}"
        )


@cli.command()
@click.argument("goal")
@click.option("--profile", "-p", help="Agent profile to use")
@click.option("--hitl/--no-hitl", default=False, help="Enable Human-in-the-Loop")
@click.option("--stream/--no-stream", default=False, help="Stream execution progress")
@click.option("--max-steps", type=int, default=50, help="Maximum execution steps")
@click.pass_context
def execute(
    ctx: click.Context, goal: str, profile: str | None, hitl: bool, stream: bool, max_steps: int
) -> None:
    """Execute any task using ROMA's intelligent agent system"""
    profile = profile or ctx.obj["profile"]
    system_manager = ctx.obj["system_manager"]

    click.echo(f"🚀 Executing task: {click.style(goal, fg='cyan', bold=True)}")
    click.echo(f"📋 Profile: {click.style(profile, fg='yellow')}")

    if hitl:
        click.echo("🔄 Human-in-the-Loop: Enabled")

    async def run_execution() -> None:
        try:
            # Initialize SystemManager
            await system_manager.initialize(profile)

            if ctx.obj["verbose"]:
                system_info = system_manager.get_system_info()
                click.echo(f"🔧 System initialized: {system_info['status']}")

            # Execute goal
            execution_options = {"enable_hitl": hitl, "max_steps": max_steps}

            result = await system_manager.execute_task(goal, **execution_options)

            # Display results
            click.echo(f"✅ Status: {click.style(str(result['status']), fg='green')}")
            click.echo(f"📝 Result: {click.style(result['result'], fg='green')}")
            click.echo(f"⏱️  Execution time: {result['execution_time']:.2f}s")
            click.echo(f"🔢 Nodes processed: {result['node_count']}")
            click.echo(f"⚙️  Framework: {result['framework']}")

            if result.get("artifacts"):
                click.echo(f"📁 Artifacts: {len(result['artifacts'])} files stored")

            await system_manager.shutdown()
            return result

        except Exception as e:
            click.echo(f"❌ Error: {click.style(str(e), fg='red')}", err=True)
            raise click.ClickException(str(e)) from e

    try:
        if stream:
            click.echo("📡 Streaming execution...")
            # TODO: Implement streaming with SystemManager
            asyncio.run(run_execution())
        else:
            with click.progressbar([1], label="Executing") as bar:
                asyncio.run(run_execution())
                bar.update(1)

    except Exception as e:
        click.echo(f"❌ Execution failed: {click.style(str(e), fg='red')}", err=True)
        raise click.ClickException(str(e)) from e


@cli.command()
@click.argument("topic")
@click.option("--profile", "-p", default="deep_research_agent", help="Research profile")
@click.option("--hitl/--no-hitl", default=False, help="Enable Human-in-the-Loop")
@click.pass_context
def research(_ctx: click.Context, topic: str, profile: str, hitl: bool) -> None:
    """Quick research command"""
    click.echo(f"🔍 Researching: {click.style(topic, fg='cyan', bold=True)}")
    click.echo(f"📋 Profile: {click.style(profile, fg='yellow')}")

    try:
        with click.progressbar([1], label="Researching") as bar:
            result = quick_research(topic, enable_hitl=hitl, profile_name=profile)
            bar.update(1)

        click.echo("📊 Research Result:")
        click.echo(click.style(result, fg="green"))

    except Exception as e:
        click.echo(f"❌ Research failed: {click.style(str(e), fg='red')}", err=True)
        raise click.ClickException(str(e)) from e


@cli.command()
@click.argument("goal")
@click.option("--max-steps", type=int, default=50, help="Maximum execution steps")
@click.option("--save-state/--no-save-state", default=False, help="Save execution state")
@click.pass_context
def async_execute(_ctx: click.Context, goal: str, max_steps: int, save_state: bool) -> None:
    """High-performance async execution"""
    click.echo(f"⚡ Async executing: {click.style(goal, fg='cyan', bold=True)}")

    async def run_async() -> dict[str, Any]:
        agent = LightweightSentientAgent.create_with_profile("general_agent")
        result = await agent.execute(goal, max_steps=max_steps, save_state=save_state)
        return result

    try:
        with click.progressbar([1], label="Async execution") as bar:
            result = asyncio.run(run_async())
            bar.update(1)

        click.echo(f"✅ Async Result: {click.style(result['final_output'], fg='green')}")
        click.echo(f"⏱️  Execution time: {result['execution_time']}s")

    except Exception as e:
        click.echo(f"❌ Async execution failed: {click.style(str(e), fg='red')}", err=True)
        raise click.ClickException(str(e)) from e


@cli.command()
@click.pass_context
def profiles(ctx: click.Context) -> None:
    """List available agent profiles"""
    click.echo("📋 Available profiles:")

    try:
        available_profiles = list_available_profiles()
        for profile in available_profiles:
            status = "✅" if profile == ctx.obj["profile"] else "  "
            click.echo(f"{status} {click.style(profile, fg='yellow')}")

        click.echo(
            f"\n🎯 Current profile: {click.style(ctx.obj['profile'], fg='green', bold=True)}"
        )

    except Exception as e:
        click.echo(f"❌ Failed to list profiles: {click.style(str(e), fg='red')}", err=True)


@cli.command()
@click.pass_context
def info(_ctx: click.Context) -> None:
    """Show system information"""
    try:
        agent = SentientAgent.create()
        system_info = agent.get_system_info()
        config_validation = agent.validate_configuration()

        click.echo(f"🚀 {system_info['name']} v{system_info['version']}")
        click.echo(f"📝 {system_info['description']}")
        click.echo(f"🌍 Environment: {click.style(system_info['environment'], fg='yellow')}")
        click.echo(f"📋 Profile: {click.style(system_info['profile'], fg='yellow')}")
        click.echo(f"💾 Cache: {'✅ Enabled' if system_info['cache_enabled'] else '❌ Disabled'}")
        click.echo(f"✅ Status: {click.style(system_info['status'].upper(), fg='green')}")

        if config_validation["valid"]:
            click.echo(f"🔧 Configuration: {click.style('Valid', fg='green')}")
        else:
            click.echo(f"⚠️  Configuration: {click.style('Issues found', fg='yellow')}")
            for issue_type, issues in config_validation["issues"].items():
                click.echo(f"   • {issue_type}: {', '.join(issues)}")

    except Exception as e:
        click.echo(f"❌ Failed to get system info: {click.style(str(e), fg='red')}", err=True)


@cli.command()
@click.pass_context
def validate(ctx: click.Context) -> None:
    """Validate configuration"""
    try:
        agent = SentientAgent.create(config_path=ctx.obj["config"])
        validation = agent.validate_configuration()

        if validation["valid"]:
            click.echo(f"✅ Configuration is {click.style('valid', fg='green')}")
        else:
            click.echo(f"⚠️  Configuration has {click.style('issues', fg='yellow')}:")
            for issue_type, issues in validation["issues"].items():
                click.echo(f"  • {click.style(issue_type, fg='red')}: {', '.join(issues)}")

        click.echo(f"📋 Profile: {validation['profile']}")
        click.echo(f"🔧 Status: {validation['status']}")

    except Exception as e:
        click.echo(f"❌ Validation failed: {click.style(str(e), fg='red')}", err=True)
        raise click.ClickException(str(e)) from e


if __name__ == "__main__":
    cli()
