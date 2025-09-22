"""
Agent Response Models for ROMA v2.0

Pydantic response models for structured agent outputs.
These models are used with Agno's native response_model parameter.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Union
from roma.domain.value_objects.task_type import TaskType


class AtomizerResult(BaseModel):
    """
    Atomizer decision result.

    Determines whether a task should be executed atomically or decomposed into subtasks.
    """
    model_config = {"frozen": True, "validate_assignment": True}

    is_atomic: bool = Field(
        ...,
        description="True if task can be executed directly by a single agent without decomposition. False if task requires breaking down into subtasks."
    )
    reasoning: str = Field(
        ...,
        min_length=1,
        description="Detailed explanation of the atomization decision, including specific factors that led to atomic vs composite classification"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score in the atomization decision (0.0 = uncertain, 1.0 = highly confident)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional decision context such as complexity metrics, estimated subtask count, or decision factors"
    )


    @classmethod
    def get_examples(cls) -> List[Dict[str, Any]]:
        """Get example AtomizerResult objects for templates."""
        return [
            {
                "is_atomic": False,
                "reasoning": "This task requires multiple steps including data gathering and analysis, indicating it needs decomposition into focused subtasks.",
                "confidence": 0.9,
                "metadata": {"complexity_score": "high", "estimated_subtasks": 4}
            },
            {
                "is_atomic": True,
                "reasoning": "This is a specific fact lookup that can be completed with a single direct query.",
                "confidence": 1.0
            },
            {
                "is_atomic": False,
                "reasoning": "The task involves multiple related concepts that need individual research and synthesis, requiring decomposition.",
                "confidence": 0.85,
                "metadata": {"complexity_score": "medium", "estimated_subtasks": 3}
            },
            {
                "is_atomic": True,
                "reasoning": "This is a single calculation with all required data provided and clear parameters specified.",
                "confidence": 0.95
            },
            {
                "is_atomic": False,
                "reasoning": "This comprehensive analysis task spans multiple domains and requires extensive research, indicating clear need for decomposition.",
                "confidence": 0.92,
                "metadata": {"complexity_score": "very_high", "estimated_subtasks": 6}
            }
        ]


class SubTask(BaseModel):
    """
    Individual subtask in a decomposition plan.
    """
    model_config = {"frozen": True}

    goal: str = Field(
        ...,
        min_length=1,
        description="Precise, self-contained subtask objective that can be executed independently with all necessary context included"
    )
    task_type: TaskType = Field(
        ...,
        description="Type of subtask operation: RETRIEVE (data gathering), WRITE (content creation), THINK (analysis/reasoning), CODE_INTERPRET (code execution), IMAGE_GENERATION (visual creation)"
    )
    priority: int = Field(
        default=0,
        description="Execution priority where higher numbers indicate more critical tasks (0 = normal priority, higher = more important)"
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="List of subtask IDs that must complete before this task can begin execution (empty list means no dependencies)"
    )
    estimated_effort: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
        description="Relative effort estimate on 1-10 scale (1 = very simple, 5 = moderate, 10 = very complex)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional subtask context such as format requirements, data sources, output specifications, or execution hints"
    )

    @classmethod
    def get_examples(cls) -> List[Dict[str, Any]]:
        """Get example SubTask objects for templates."""
        return [
            {
                "goal": "Find current electric vehicle market share data for Tesla in the US",
                "task_type": "RETRIEVE",
                "priority": 0,
                "dependencies": [],
                "estimated_effort": 3,
                "metadata": {"data_source": "market_research", "geographic_scope": "US"}
            },
            {
                "goal": "Analyze competitive positioning based on gathered market data",
                "task_type": "THINK",
                "priority": 0,
                "dependencies": ["market_data_task"],
                "estimated_effort": 4
            },
            {
                "goal": "Generate executive summary of market analysis findings",
                "task_type": "WRITE",
                "priority": 1,
                "dependencies": ["analysis_task"],
                "estimated_effort": 2,
                "metadata": {"format": "executive_summary", "max_length": 500}
            },
            {
                "goal": "Process quarterly sales data and calculate growth metrics",
                "task_type": "CODE_INTERPRET",
                "priority": 0,
                "dependencies": [],
                "estimated_effort": 3,
                "metadata": {"data_format": "csv", "metrics": ["growth_rate", "market_share"]}
            },
            {
                "goal": "Create market trend visualization chart",
                "task_type": "IMAGE_GENERATION",
                "priority": 0,
                "dependencies": ["data_processing_task"],
                "estimated_effort": 2,
                "metadata": {"chart_type": "line_chart", "style": "professional"}
            }
        ]


class PlannerResult(BaseModel):
    """
    Planner decomposition result.

    Contains the breakdown of a complex task into executable subtasks.
    """
    model_config = {"frozen": True}

    subtasks: List[SubTask] = Field(
        ...,
        min_items=1,
        description="List of 3-6 planned subtasks that collectively accomplish the parent goal, each with specific objectives and clear task types"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Detailed planning rationale explaining decomposition approach and dependency decisions"
    )
    estimated_total_effort: Optional[int] = Field(
        default=None,
        ge=1,
        description="Sum of estimated effort across all subtasks, representing total work required to complete the parent goal"
    )

    @classmethod
    def get_examples(cls) -> List[Dict[str, Any]]:
        """Get example PlannerResult objects for templates."""
        return [
            {
                "subtasks": [
                    {
                        "goal": "Find current Tesla market share in electric vehicle segment",
                        "task_type": "RETRIEVE",
                        "priority": 0,
                        "dependencies": [],
                        "estimated_effort": 3
                    },
                    {
                        "goal": "Analyze Tesla's competitive advantages in the EV market",
                        "task_type": "THINK",
                        "priority": 0,
                        "dependencies": [],
                        "estimated_effort": 4
                    }
                ],
                "reasoning": "Independent data gathering and analysis can be performed simultaneously for efficiency.",
                "estimated_total_effort": 7
            },
            {
                "subtasks": [
                    {
                        "goal": "Gather Q3 financial data including revenue and expenses",
                        "task_type": "RETRIEVE",
                        "priority": 0,
                        "dependencies": [],
                        "estimated_effort": 2
                    },
                    {
                        "goal": "Analyze financial trends and calculate key performance metrics",
                        "task_type": "THINK",
                        "priority": 0,
                        "dependencies": ["financial_data"],
                        "estimated_effort": 3
                    },
                    {
                        "goal": "Create comprehensive quarterly financial report",
                        "task_type": "WRITE",
                        "priority": 0,
                        "dependencies": ["financial_analysis"],
                        "estimated_effort": 4
                    }
                ],
                "reasoning": "Financial analysis requires data gathering first, then report creation needs completed analysis.",
                "estimated_total_effort": 9
            },
            {
                "subtasks": [
                    {
                        "goal": "Load customer survey dataset and validate data quality",
                        "task_type": "CODE_INTERPRET",
                        "priority": 0,
                        "dependencies": [],
                        "estimated_effort": 2
                    },
                    {
                        "goal": "Perform statistical analysis on survey responses",
                        "task_type": "CODE_INTERPRET",
                        "priority": 0,
                        "dependencies": ["data_validation"],
                        "estimated_effort": 4
                    },
                    {
                        "goal": "Generate visualization charts for key survey findings",
                        "task_type": "IMAGE_GENERATION",
                        "priority": 0,
                        "dependencies": ["statistical_analysis"],
                        "estimated_effort": 3
                    }
                ],
                "reasoning": "Data processing pipeline requires sequential execution for data dependencies.",
                "estimated_total_effort": 9
            },
            {
                "subtasks": [
                    {
                        "goal": "Research renewable energy adoption rates by country",
                        "task_type": "RETRIEVE",
                        "priority": 0,
                        "dependencies": [],
                        "estimated_effort": 3
                    },
                    {
                        "goal": "Find government policies supporting renewable energy",
                        "task_type": "RETRIEVE",
                        "priority": 0,
                        "dependencies": [],
                        "estimated_effort": 3
                    },
                    {
                        "goal": "Identify technology trends in renewable energy sector",
                        "task_type": "RETRIEVE",
                        "priority": 0,
                        "dependencies": [],
                        "estimated_effort": 3
                    },
                    {
                        "goal": "Synthesize renewable energy landscape analysis",
                        "task_type": "WRITE",
                        "priority": 0,
                        "dependencies": ["adoption_research", "policy_research", "technology_research"],
                        "estimated_effort": 5
                    }
                ],
                "reasoning": "Multiple parallel research streams feeding into final synthesis.",
                "estimated_total_effort": 14
            },
            {
                "subtasks": [
                    {
                        "goal": "Define visual brand elements and style guidelines",
                        "task_type": "THINK",
                        "priority": 0,
                        "dependencies": [],
                        "estimated_effort": 3
                    },
                    {
                        "goal": "Create primary logo design",
                        "task_type": "IMAGE_GENERATION",
                        "priority": 0,
                        "dependencies": ["brand_guidelines"],
                        "estimated_effort": 4
                    },
                    {
                        "goal": "Generate supporting marketing materials",
                        "task_type": "IMAGE_GENERATION",
                        "priority": 0,
                        "dependencies": ["logo_design"],
                        "estimated_effort": 5
                    }
                ],
                "reasoning": "Brand development requires sequential steps to maintain consistency across visual elements.",
                "estimated_total_effort": 12
            }
        ]


class ExecutorResult(BaseModel):
    """
    Executor execution result.

    Contains the output of atomic task execution.
    """
    model_config = {"frozen": True}

    result: Any = Field(
        ...,
        description="Primary execution result containing the actual output, findings, or generated content from task completion"
    )
    sources: List[str] = Field(
        default_factory=list,
        description="List of information sources, references, or data origins used during task execution (URLs, documents, databases, etc.)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Execution metadata including method details, parameters used, intermediate results, or processing information"
    )
    success: bool = Field(
        default=True,
        description="Boolean indicating whether task execution completed successfully (True) or encountered errors (False)"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score in result accuracy and completeness (0.0 = unreliable, 1.0 = highly confident)"
    )
    tokens_used: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of language model tokens consumed during execution (for cost tracking and optimization)"
    )
    execution_time: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Total execution time in seconds from task start to completion (for performance monitoring)"
    )

    @classmethod
    def get_examples(cls) -> List[Dict[str, Any]]:
        """Get example ExecutorResult objects for templates."""
        return [
            {
                "result": "Tesla currently holds approximately 17.8% of the global electric vehicle market share as of Q3 2024, representing a slight decline from 18.4% in Q2 2024 due to increased competition from BYD and other manufacturers.",
                "sources": [
                    "https://www.ev-volumes.com/country/total-world-plug-in-vehicle-volumes/",
                    "https://insideevs.com/news/692847/global-electric-car-sales-q3-2024/",
                    "https://www.canalys.com/newsroom/global-electric-vehicle-market-q3-2024"
                ],
                "metadata": {
                    "search_queries": ["Tesla market share 2024", "global EV market Q3 2024"],
                    "data_sources": ["market_research", "industry_reports"],
                    "geographic_scope": "global"
                },
                "success": True,
                "confidence": 0.92,
                "tokens_used": 1250,
                "execution_time": 3.8
            },
            {
                "result": "# Competitive Analysis: Tesla vs Traditional Automakers\n\n## Key Advantages\n- **Technology Leadership**: Advanced battery technology and software integration\n- **Charging Infrastructure**: Extensive Supercharger network with 50,000+ stations globally\n- **Brand Positioning**: Premium electric-first brand with strong consumer loyalty\n\n## Strategic Challenges\n- Increasing competition from legacy automakers (Ford, GM, VW)\n- Price pressure from Chinese manufacturers (BYD, NIO)\n- Supply chain dependencies for battery materials",
                "sources": [
                    "https://www.mckinsey.com/industries/automotive-and-assembly/our-insights/the-ev-tipping-point",
                    "https://www.bloomberg.com/news/articles/2024-10-15/tesla-faces-growing-competition-in-ev-market"
                ],
                "metadata": {
                    "analysis_framework": "competitive_positioning",
                    "format": "markdown_report",
                    "word_count": 185
                },
                "success": True,
                "confidence": 0.88,
                "tokens_used": 2100,
                "execution_time": 5.2
            },
            {
                "result": {
                    "revenue_growth_rate": 0.127,
                    "quarterly_data": [
                        {"quarter": "Q1_2024", "revenue": 145.2, "growth": 0.089},
                        {"quarter": "Q2_2024", "revenue": 158.7, "growth": 0.093},
                        {"quarter": "Q3_2024", "revenue": 168.4, "growth": 0.061},
                        {"quarter": "Q4_2024", "revenue": 189.8, "growth": 0.127}
                    ],
                    "statistical_summary": {
                        "mean_growth": 0.0925,
                        "std_deviation": 0.0285,
                        "trend": "accelerating_growth"
                    }
                },
                "sources": ["internal_financial_database"],
                "metadata": {
                    "calculation_method": "compound_annual_growth_rate",
                    "data_period": "2024_full_year",
                    "currency": "USD_billions"
                },
                "success": True,
                "confidence": 0.98,
                "tokens_used": 450,
                "execution_time": 1.2
            },
            {
                "result": "import pandas as pd\nimport matplotlib.pyplot as plt\nimport numpy as np\n\n# Load and process survey data\ndf = pd.read_csv('customer_survey_2024.csv')\ndf_clean = df.dropna(subset=['satisfaction_score', 'recommendation_score'])\n\n# Statistical analysis\ncorrelation = df_clean['satisfaction_score'].corr(df_clean['recommendation_score'])\nmean_satisfaction = df_clean['satisfaction_score'].mean()\n\n# Generate visualization\nplt.figure(figsize=(10, 6))\nplt.scatter(df_clean['satisfaction_score'], df_clean['recommendation_score'], alpha=0.6)\nplt.xlabel('Customer Satisfaction Score')\nplt.ylabel('Net Promoter Score')\nplt.title('Customer Satisfaction vs Recommendation Correlation')\nplt.savefig('satisfaction_analysis.png')\n\nprint(f'Correlation coefficient: {correlation:.3f}')\nprint(f'Average satisfaction: {mean_satisfaction:.2f}/10')",
                "sources": ["customer_survey_2024.csv"],
                "metadata": {
                    "code_language": "python",
                    "libraries_used": ["pandas", "matplotlib", "numpy"],
                    "output_files": ["satisfaction_analysis.png"],
                    "data_points_processed": 1247
                },
                "success": True,
                "confidence": 0.95,
                "execution_time": 2.1
            },
            {
                "result": "Task execution failed due to API rate limiting. The external data source (Financial API) returned HTTP 429 status after 3 retry attempts. Partial data was retrieved for Q1-Q2 2024 before the rate limit was hit.",
                "sources": ["https://api.financialdata.com/v2/company-metrics"],
                "metadata": {
                    "error_type": "rate_limit_exceeded",
                    "retry_attempts": 3,
                    "partial_data_available": True,
                    "suggested_retry_time": "2024-11-15T16:30:00Z"
                },
                "success": False,
                "confidence": 0.0,
                "tokens_used": 120,
                "execution_time": 8.7
            }
        ]


class AggregatorResult(BaseModel):
    """
    Aggregator synthesis result.

    Contains the synthesis of multiple subtask results into a cohesive output.
    """
    model_config = {"frozen": True}

    synthesized_result: str = Field(
        ...,
        min_length=1,
        description="Final comprehensive output that synthesizes and integrates all subtask results into a cohesive, complete response"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Overall confidence in synthesis quality and completeness (0.0 = uncertain, 1.0 = highly confident)"
    )
    sources_used: List[str] = Field(
        default_factory=list,
        description="List of subtask IDs whose results were incorporated into the final synthesis"
    )
    gaps_identified: List[str] = Field(
        default_factory=list,
        description="List of information gaps, missing data, or areas where subtask results were incomplete or contradictory"
    )
    quality_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Assessment of output quality based on completeness, coherence, and accuracy (0.0 = poor, 1.0 = excellent)"
    )

    @classmethod
    def get_examples(cls) -> List[Dict[str, Any]]:
        """Get example AggregatorResult objects for templates."""
        return [
            {
                "synthesized_result": "# Tesla Electric Vehicle Market Analysis\n\nBased on comprehensive research across market data, competitive landscape, and financial performance, Tesla maintains a strong but evolving position in the global EV market. **Market Share**: Tesla holds 17.8% of global EV sales as of Q3 2024, down slightly from 18.4% in Q2 due to increased competition. **Competitive Advantages**: Tesla's technological leadership in battery efficiency, autonomous driving capabilities, and charging infrastructure (50,000+ Supercharger stations) provides sustainable differentiation. **Financial Performance**: Strong revenue growth of 12.7% year-over-year with improving profit margins. **Key Challenges**: Intensifying competition from Chinese manufacturers (BYD, NIO) and legacy automakers (Ford, VW) transitioning to electric. **Future Outlook**: Tesla's focus on affordable models and expanding production capacity positions it well for continued market leadership despite increasing competitive pressure.",
                "confidence": 0.89,
                "sources_used": ["market_data_task", "competitive_analysis_task", "financial_performance_task"],
                "gaps_identified": [],
                "quality_score": 0.92
            },
            {
                "synthesized_result": "# Q3 2024 Financial Performance Report\n\n## Executive Summary\nStrong quarterly performance with revenue growth exceeding targets and improved operational efficiency across all business units.\n\n## Financial Highlights\n- **Revenue**: $189.8B (+12.7% YoY)\n- **Operating Margin**: 24.3% (+2.1pp improvement)\n- **Net Income**: $45.2B (+18.5% YoY)\n\n## Operational Achievements\n- Successfully launched three new product lines\n- Expanded into European markets with 15% market penetration\n- Achieved 98.5% customer satisfaction scores\n\n## Market Context\nOutperformed industry benchmarks despite challenging economic conditions including supply chain disruptions and increased material costs.\n\n## Strategic Outlook\nPositioned for continued growth with strong pipeline of innovations and expanding market presence.",
                "confidence": 0.95,
                "sources_used": ["financial_data_task", "operational_highlights_task", "market_context_task"],
                "gaps_identified": ["competitor_response_analysis"],
                "quality_score": 0.94
            },
            {
                "synthesized_result": "# Customer Survey Analysis: Satisfaction and Loyalty Insights\n\n## Key Findings\nComprehensive analysis of 1,247 customer responses reveals strong positive correlation (r=0.847) between satisfaction scores and recommendation likelihood, indicating robust customer loyalty foundation.\n\n## Statistical Summary\n- **Average Satisfaction**: 8.2/10 (+0.3 from previous quarter)\n- **Net Promoter Score**: 67 (Industry benchmark: 45)\n- **Customer Retention**: 94.3% (+2.1pp improvement)\n\n## Correlation Analysis\nStrong positive relationship between satisfaction and advocacy demonstrates that investment in customer experience directly translates to word-of-mouth marketing value.\n\n## Segment Analysis\n- Premium customers: 9.1/10 satisfaction\n- Standard customers: 7.8/10 satisfaction\n- Enterprise customers: 8.7/10 satisfaction\n\n## Recommendations\n1. Maintain current service quality standards\n2. Focus improvement efforts on standard customer segment\n3. Leverage high NPS for referral programs",
                "confidence": 0.96,
                "sources_used": ["data_processing_task", "statistical_analysis_task", "visualization_task"],
                "gaps_identified": [],
                "quality_score": 0.93
            },
            {
                "synthesized_result": "# Renewable Energy Landscape Analysis\n\n## Market Dynamics\nGlobal renewable energy adoption accelerating with wind and solar leading growth. **Adoption Rates**: Solar capacity increased 73% globally in 2024, wind power up 45%. Key markets include China (40% of global capacity), USA (15%), and EU (18%).\n\n## Policy Environment\n**Supportive Frameworks**: 89 countries have net-zero commitments with policy mechanisms including feed-in tariffs, renewable portfolio standards, and carbon pricing. **Investment Incentives**: $2.8T in planned renewable investments through 2030.\n\n## Technology Trends\n**Innovation Areas**: Advanced battery storage (cost down 85% since 2020), floating solar farms, offshore wind turbines (15MW+ capacity), and green hydrogen production. **Efficiency Gains**: Solar panel efficiency reached 26.1% in commercial applications.\n\n## Market Outlook\nRenewable energy projected to comprise 85% of new power generation capacity through 2030, driven by cost competitiveness and policy support. Storage solutions critical for grid stability and intermittency management.",
                "confidence": 0.91,
                "sources_used": ["adoption_research_task", "policy_research_task", "technology_research_task"],
                "gaps_identified": ["regional_regulatory_variations", "grid_integration_challenges"],
                "quality_score": 0.88
            },
            {
                "synthesized_result": "# Brand Identity Development Strategy\n\n## Visual Identity Framework\nComprehensive brand system established with modern, minimalist aesthetic emphasizing innovation and reliability. **Color Palette**: Primary blue (#1E3A8A) representing trust, secondary green (#059669) for growth, neutral grays for balance. **Typography**: Custom sans-serif font family optimizing readability across digital and print media.\n\n## Logo Design\nCreated distinctive mark combining geometric precision with organic flow, symbolizing technological advancement harmonized with human-centered design. Responsive logo system adapts across applications from business cards to billboard scale.\n\n## Marketing Applications\nDeveloped complete brand ecosystem including business stationery, digital templates, presentation formats, and packaging guidelines. **Consistency Framework**: 47-page brand guidelines ensuring consistent application across all touchpoints.\n\n## Implementation Roadmap\nPhased rollout over 6 months: digital assets (Month 1-2), print materials (Month 3-4), environmental applications (Month 5-6). Training program for 12 departments on brand standards and application guidelines.",
                "confidence": 0.87,
                "sources_used": ["brand_guidelines_task", "logo_design_task", "marketing_materials_task"],
                "gaps_identified": ["user_testing_feedback", "competitive_differentiation_analysis"],
                "quality_score": 0.85
            }
        ]


class PlanModifierResult(BaseModel):
    """
    Plan modification result.

    Contains modifications to an existing plan based on feedback or new information.
    """
    model_config = {"frozen": True}

    modified_subtasks: List[SubTask] = Field(
        ...,
        description="Complete updated list of subtasks after applying modifications, including new, changed, and unchanged tasks"
    )
    changes_made: List[str] = Field(
        ...,
        min_items=1,
        description="Detailed list of specific changes applied to the plan (additions, removals, modifications, reordering)"
    )
    reasoning: str = Field(
        ...,
        min_length=1,
        description="Comprehensive rationale explaining why modifications were necessary and how they address identified issues or feedback"
    )
    impact_assessment: Optional[str] = Field(
        default=None,
        description="Analysis of how changes affect execution timeline, resource requirements, dependencies, and overall plan success probability"
    )
    new_dependencies: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description="Updated dependency graph reflecting any new or changed task relationships after plan modifications"
    )

    @classmethod
    def get_examples(cls) -> List[Dict[str, Any]]:
        """Get example PlanModifierResult objects for templates."""
        return [
            {
                "modified_subtasks": [
                    {
                        "goal": "Research Tesla's current global market share in electric vehicles with detailed regional breakdown",
                        "task_type": "RETRIEVE",
                        "priority": 1,
                        "dependencies": [],
                        "estimated_effort": 4,
                        "metadata": {"geographic_scope": "global", "breakdown_level": "regional"}
                    },
                    {
                        "goal": "Analyze Tesla's competitive advantages including technology, infrastructure, and brand positioning",
                        "task_type": "THINK",
                        "priority": 0,
                        "dependencies": ["market_share_task"],
                        "estimated_effort": 3
                    },
                    {
                        "goal": "Evaluate Tesla's financial performance trends over the past 2 years with key metrics",
                        "task_type": "RETRIEVE",
                        "priority": 0,
                        "dependencies": [],
                        "estimated_effort": 3,
                        "metadata": {"time_period": "2_years", "metrics": ["revenue", "profit_margin", "deliveries"]}
                    }
                ],
                "changes_made": [
                    "Increased priority of market share research from 0 to 1 based on feedback",
                    "Added regional breakdown requirement to market share task goal",
                    "Added dependency from competitive analysis to market share task",
                    "Extended financial performance analysis from 1 year to 2 years",
                    "Added specific metrics requirement to financial analysis"
                ],
                "reasoning": "Human feedback indicated that market share data is critical foundation for competitive analysis, requiring higher priority and regional detail. The competitive analysis should build on market data rather than running in parallel. Financial analysis needs broader historical context to identify meaningful trends.",
                "impact_assessment": "Changes create sequential dependency that may increase total execution time by 1-2 hours but significantly improve analysis quality. Resource requirements remain unchanged. Success probability improved from 0.8 to 0.9 due to better task sequencing.",
                "new_dependencies": {
                    "competitive_analysis_task": ["market_share_task"]
                }
            },
            {
                "modified_subtasks": [
                    {
                        "goal": "Gather Q3 2024 financial data including revenue, expenses, cash flow, and segment performance",
                        "task_type": "RETRIEVE",
                        "priority": 0,
                        "dependencies": [],
                        "estimated_effort": 3,
                        "metadata": {"data_sources": ["financial_statements", "earnings_reports"]}
                    },
                    {
                        "goal": "Research external market conditions and competitive landscape during Q3 2024",
                        "task_type": "RETRIEVE",
                        "priority": 0,
                        "dependencies": [],
                        "estimated_effort": 2
                    },
                    {
                        "goal": "Analyze Q3 performance trends with variance analysis and key driver identification",
                        "task_type": "THINK",
                        "priority": 0,
                        "dependencies": ["financial_data_task", "market_conditions_task"],
                        "estimated_effort": 4
                    },
                    {
                        "goal": "Create comprehensive quarterly report with executive summary and detailed analysis sections",
                        "task_type": "WRITE",
                        "priority": 0,
                        "dependencies": ["trend_analysis_task"],
                        "estimated_effort": 5,
                        "metadata": {"format": "formal_report", "sections": ["executive_summary", "financial_analysis", "market_context"]}
                    }
                ],
                "changes_made": [
                    "Added market conditions research as separate parallel task",
                    "Enhanced financial data gathering to include segment performance",
                    "Made trend analysis dependent on both financial data and market conditions",
                    "Specified report format and required sections in final task",
                    "Increased effort estimate for analysis task from 3 to 4"
                ],
                "reasoning": "User feedback emphasized need for external market context to properly interpret financial performance. Separate market research task enables parallel execution while providing richer analytical foundation. Report structure clarification ensures deliverable meets expectations.",
                "impact_assessment": "Addition of market research increases parallel capacity utilization and provides better analytical depth. Total estimated effort increases from 9 to 11 hours but delivers significantly higher quality output. Timeline remains similar due to parallel execution.",
                "new_dependencies": {
                    "trend_analysis_task": ["financial_data_task", "market_conditions_task"]
                }
            },
            {
                "modified_subtasks": [
                    {
                        "goal": "Load customer survey dataset, validate data quality, and clean missing values",
                        "task_type": "CODE_INTERPRET",
                        "priority": 0,
                        "dependencies": [],
                        "estimated_effort": 3,
                        "metadata": {"validation_steps": ["completeness", "format", "outliers"]}
                    },
                    {
                        "goal": "Perform descriptive statistics and correlation analysis on satisfaction metrics",
                        "task_type": "CODE_INTERPRET",
                        "priority": 0,
                        "dependencies": ["data_validation_task"],
                        "estimated_effort": 3
                    },
                    {
                        "goal": "Conduct customer segmentation analysis based on demographics and behavior patterns",
                        "task_type": "CODE_INTERPRET",
                        "priority": 0,
                        "dependencies": ["statistical_analysis_task"],
                        "estimated_effort": 4
                    },
                    {
                        "goal": "Generate comprehensive visualizations including correlation plots and segment comparisons",
                        "task_type": "CODE_INTERPRET",
                        "priority": 0,
                        "dependencies": ["segmentation_analysis_task"],
                        "estimated_effort": 3
                    }
                ],
                "changes_made": [
                    "Added customer segmentation analysis as new subtask",
                    "Made segmentation dependent on statistical analysis",
                    "Updated visualization task to include segment comparisons",
                    "Removed generic report writing task in favor of code-based analysis",
                    "Created linear dependency chain for data processing pipeline"
                ],
                "reasoning": "Human feedback requested deeper customer insights through segmentation analysis. This requires processing statistical results before segmentation, making sequential dependencies necessary. Visualization enhancement provides richer insights into customer patterns.",
                "impact_assessment": "Sequential processing increases execution time but enables more sophisticated analysis. Total effort increases from 8 to 13 hours. Higher complexity but significantly more valuable insights for business decisions.",
                "new_dependencies": {
                    "segmentation_analysis_task": ["statistical_analysis_task"],
                    "visualization_task": ["segmentation_analysis_task"]
                }
            },
            {
                "modified_subtasks": [
                    {
                        "goal": "Research current renewable energy adoption rates by technology type and geographic region",
                        "task_type": "RETRIEVE",
                        "priority": 0,
                        "dependencies": [],
                        "estimated_effort": 4,
                        "metadata": {"technologies": ["solar", "wind", "hydro", "geothermal"], "scope": "global"}
                    },
                    {
                        "goal": "Identify key government policies and incentive programs supporting renewable energy transition",
                        "task_type": "RETRIEVE",
                        "priority": 0,
                        "dependencies": [],
                        "estimated_effort": 3
                    },
                    {
                        "goal": "Analyze emerging technology trends and cost reduction trajectories in renewable energy",
                        "task_type": "RETRIEVE",
                        "priority": 0,
                        "dependencies": [],
                        "estimated_effort": 3
                    },
                    {
                        "goal": "Examine investment flows and financing mechanisms in renewable energy sector",
                        "task_type": "RETRIEVE",
                        "priority": 0,
                        "dependencies": [],
                        "estimated_effort": 3
                    },
                    {
                        "goal": "Synthesize comprehensive renewable energy landscape analysis with future projections",
                        "task_type": "WRITE",
                        "priority": 1,
                        "dependencies": ["adoption_research_task", "policy_research_task", "technology_research_task", "investment_research_task"],
                        "estimated_effort": 6
                    }
                ],
                "changes_made": [
                    "Added investment flows research as fourth parallel research stream",
                    "Enhanced adoption research to include technology breakdown",
                    "Increased synthesis task priority to 1",
                    "Made synthesis dependent on all four research tasks",
                    "Increased synthesis effort estimate from 5 to 6 hours"
                ],
                "reasoning": "User emphasized importance of financial aspects in renewable energy analysis. Investment research provides critical economic context. Technology breakdown in adoption research enables more nuanced analysis. Higher priority on synthesis ensures comprehensive integration.",
                "impact_assessment": "Additional research stream maintains parallel execution while providing broader analytical foundation. Total effort increases from 14 to 19 hours but delivers significantly more comprehensive analysis covering all key aspects of renewable energy landscape.",
                "new_dependencies": {
                    "synthesis_task": ["adoption_research_task", "policy_research_task", "technology_research_task", "investment_research_task"]
                }
            },
            {
                "modified_subtasks": [
                    {
                        "goal": "Define comprehensive visual brand guidelines including color psychology and accessibility standards",
                        "task_type": "THINK",
                        "priority": 0,
                        "dependencies": [],
                        "estimated_effort": 4,
                        "metadata": {"accessibility": "WCAG_2.1", "psychology": "color_theory"}
                    },
                    {
                        "goal": "Create primary logo design with multiple format variations and usage specifications",
                        "task_type": "IMAGE_GENERATION",
                        "priority": 0,
                        "dependencies": ["brand_guidelines_task"],
                        "estimated_effort": 5,
                        "metadata": {"formats": ["svg", "png", "pdf"], "variations": ["horizontal", "stacked", "icon"]}
                    },
                    {
                        "goal": "Develop secondary brand elements including icons, patterns, and graphic elements",
                        "task_type": "IMAGE_GENERATION",
                        "priority": 0,
                        "dependencies": ["logo_design_task"],
                        "estimated_effort": 4
                    },
                    {
                        "goal": "Generate comprehensive marketing material templates and brand applications",
                        "task_type": "IMAGE_GENERATION",
                        "priority": 0,
                        "dependencies": ["secondary_elements_task"],
                        "estimated_effort": 6,
                        "metadata": {"templates": ["business_cards", "letterhead", "presentations", "social_media"]}
                    }
                ],
                "changes_made": [
                    "Enhanced brand guidelines to include accessibility standards and color psychology",
                    "Added secondary brand elements as intermediate development step",
                    "Specified multiple logo format variations and usage rules",
                    "Expanded marketing materials to include social media templates",
                    "Created linear dependency chain ensuring design consistency"
                ],
                "reasoning": "Client feedback emphasized need for comprehensive brand system with accessibility compliance. Secondary elements step ensures visual cohesion before final applications. Social media templates are essential for modern brand deployment.",
                "impact_assessment": "Enhanced brand development creates more robust foundation and broader application range. Sequential dependencies ensure consistency but increase timeline. Total effort rises from 12 to 19 hours but delivers enterprise-grade brand system.",
                "new_dependencies": {
                    "secondary_elements_task": ["logo_design_task"],
                    "marketing_materials_task": ["secondary_elements_task"]
                }
            }
        ]


# Convenience functions for creating results
def create_atomic_result(reasoning: str, confidence: float = 1.0) -> AtomizerResult:
    """Create an atomic (EXECUTE) atomizer result."""
    return AtomizerResult(
        is_atomic=True,
        reasoning=reasoning,
        confidence=confidence
    )


def create_composite_result(reasoning: str, confidence: float = 1.0) -> AtomizerResult:
    """Create a composite (PLAN) atomizer result."""
    return AtomizerResult(
        is_atomic=False,
        reasoning=reasoning,
        confidence=confidence
    )


def create_successful_execution(result: Any, sources: List[str] = None) -> ExecutorResult:
    """Create a successful executor result."""
    return ExecutorResult(
        result=result,
        sources=sources or [],
        success=True
    )


def create_failed_execution(error_msg: str, metadata: Dict[str, Any] = None) -> ExecutorResult:
    """Create a failed executor result."""
    return ExecutorResult(
        result=error_msg,
        metadata=metadata or {},
        success=False,
        confidence=0.0
    )