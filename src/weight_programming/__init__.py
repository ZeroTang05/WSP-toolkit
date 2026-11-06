"""Paper-grounded primitives for direct, CPU-side parameter programming."""
from .tensor_store import EditPolicy, WeightStore
from .directions import contrast_direction, project_output_direction, compare_edits
from .protocol import ForwardPolicy, BudgetLedger
from .forward_service import ForwardService, PracticeSet
from .reconstruction import input_hessian, sparsegpt_reconstruct
from .cpu_kernel import CPUKernel
from .programming_session import ProgrammingSession
from .patching import ActivationPatchingStudy,replace_activation
from .tool_api import tool_schemas,dispatch_tool

__all__=['EditPolicy','WeightStore','contrast_direction','project_output_direction','compare_edits',
    'ForwardPolicy','BudgetLedger','ForwardService','PracticeSet',
    'input_hessian','sparsegpt_reconstruct','CPUKernel','ProgrammingSession',
    'ActivationPatchingStudy','replace_activation','tool_schemas','dispatch_tool']
