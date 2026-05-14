# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Enhanced Error Handling Module for Inference Builder

This module provides rich error context, categorization, and troubleshooting guidance
to make debugging easier for both humans and AI tools.
"""

import traceback
import time
import uuid
import json
import threading
import inspect
import os
from dataclasses import dataclass, field, asdict
from typing import ClassVar, Dict, List, Optional, Any, Union
from enum import Enum
import numpy as np

# Try to import torch, but make it optional
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ErrorCategory(Enum):
    """Categorize errors for easier filtering and analysis"""
    CONFIGURATION = "configuration"
    DATA_VALIDATION = "data_validation"
    ASSET_NOT_FOUND = "asset_not_found"
    PROCESSING = "processing"
    BACKEND = "backend"
    TIMEOUT = "timeout"
    RESOURCE = "resource"
    NETWORK = "network"
    INTERNAL = "internal"
    CODEC = "codec"
    QUEUE = "queue"


class ErrorSeverity(Enum):
    """Error severity levels"""
    CRITICAL = "critical"  # Service-breaking
    ERROR = "error"        # Request-breaking
    WARNING = "warning"    # Degraded but continuing
    INFO = "info"          # Informational only


@dataclass
class Error:
    """
    Base Error class for simple error messages.

    This is the base class that EnhancedError inherits from.
    Use this for simple errors that don't need rich context.
    Use EnhancedError (via ErrorFactory) for structured errors with full context.
    """
    message: str

    def __bool__(self):
        """Return False so error objects evaluate to False in boolean context"""
        return False

    def __str__(self):
        """Human-readable string representation"""
        return self.message


@dataclass
class EnhancedError(Error):
    """
    Enhanced error class with rich context for AI-powered troubleshooting

    This class provides comprehensive error information including:
    - Categorization and severity
    - Contextual information (component, operation, model)
    - Technical details (stack traces, error codes)
    - Data context (inputs, expected vs actual)
    - Troubleshooting guidance and remediation steps
    - Error chaining for root cause analysis

    Inherits from Error base class, which provides:
    - message field
    - __bool__() override (returns False for boolean checks)
    """

    # Core error information (message inherited from Error)
    category: ErrorCategory
    severity: ErrorSeverity = ErrorSeverity.ERROR

    # Contextual information
    component: str = ""  # e.g., "DataFlow", "ModelOperator", "ImageInputDataFlow"
    operation: str = ""  # e.g., "bind_input", "_process_custom_data"
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    # Technical details
    error_code: str = ""  # e.g., "ERR_DF_001", "ERR_MO_005"
    stack_trace: Optional[str] = None
    exception_type: Optional[str] = None

    # Data context
    input_data: Dict[str, Any] = field(default_factory=dict)
    expected_data: Dict[str, Any] = field(default_factory=dict)
    actual_data: Dict[str, Any] = field(default_factory=dict)

    # Pipeline context
    model_name: Optional[str] = None
    dataflow_names: List[str] = field(default_factory=list)
    tensor_names: List[str] = field(default_factory=list)

    # Troubleshooting guidance
    troubleshooting_hints: List[str] = field(default_factory=list)
    related_config: Dict[str, Any] = field(default_factory=dict)
    remediation_steps: List[str] = field(default_factory=list)

    # Error chain for root cause analysis
    caused_by: Optional['EnhancedError'] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        """Human-readable string representation"""
        parts = [f"[{self.error_code or 'ERROR'}]"]
        if self.component:
            parts.append(f"{self.component}")
        if self.operation:
            parts.append(f".{self.operation}()")
        parts.append(f": {self.message}")
        return " ".join(parts)

    def to_dict(self, include_stack_trace: bool = True, sanitize_data: bool = True) -> Dict:
        """
        Convert to dictionary for structured logging

        Args:
            include_stack_trace: Whether to include full stack trace (can be verbose)
            sanitize_data: Whether to sanitize large data structures

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        result = {
            "error_id": self.request_id,
            "timestamp": self.timestamp,
            "iso_timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp)),
            "category": self.category.value,
            "severity": self.severity.value,
            "component": self.component,
            "operation": self.operation,
            "message": self.message,
            "error_code": self.error_code,
            "exception_type": self.exception_type,
        }

        if include_stack_trace and self.stack_trace:
            result["stack_trace"] = self.stack_trace

        # Add data context
        if sanitize_data:
            result["input_data"] = self._sanitize_data(self.input_data)
            result["actual_data"] = self._sanitize_data(self.actual_data)
        else:
            result["input_data"] = self.input_data
            result["actual_data"] = self.actual_data

        result["expected_data"] = self.expected_data

        # Add pipeline context
        if self.model_name:
            result["model_name"] = self.model_name
        if self.dataflow_names:
            result["dataflow_names"] = self.dataflow_names
        if self.tensor_names:
            result["tensor_names"] = self.tensor_names

        # Add troubleshooting information
        if self.troubleshooting_hints:
            result["troubleshooting_hints"] = self.troubleshooting_hints
        if self.remediation_steps:
            result["remediation_steps"] = self.remediation_steps
        if self.related_config:
            result["related_config"] = self._sanitize_config(self.related_config)

        # Add metadata
        if self.metadata:
            result["metadata"] = self.metadata

        # Add error chain
        if self.caused_by:
            result["caused_by"] = self.caused_by.to_dict(
                include_stack_trace=False,  # Don't include nested stack traces
                sanitize_data=sanitize_data
            )

        return result

    def _sanitize_data(self, data: Dict) -> Dict:
        """Sanitize large tensors/arrays for logging"""
        if not data:
            return {}

        sanitized = {}
        for k, v in data.items():
            try:
                if isinstance(v, np.ndarray):
                    sanitized[k] = {
                        "type": "numpy.ndarray",
                        "shape": str(v.shape),
                        "dtype": str(v.dtype),
                        "sample": self._safe_sample(v)
                    }
                elif TORCH_AVAILABLE and isinstance(v, torch.Tensor):
                    sanitized[k] = {
                        "type": "torch.Tensor",
                        "shape": str(list(v.shape)),
                        "dtype": str(v.dtype),
                        "device": str(v.device),
                        "sample": self._safe_sample(v.cpu().numpy() if v.is_cuda else v.numpy())
                    }
                elif isinstance(v, (list, tuple)):
                    if len(v) > 10:
                        sanitized[k] = {
                            "type": type(v).__name__,
                            "length": len(v),
                            "sample": str(v[:3]) + " ... " + str(v[-1:])
                        }
                    else:
                        sanitized[k] = v
                elif isinstance(v, dict):
                    sanitized[k] = self._sanitize_data(v)
                elif isinstance(v, str) and len(v) > 500:
                    sanitized[k] = v[:500] + "... [truncated]"
                else:
                    sanitized[k] = v
            except Exception as e:
                sanitized[k] = f"<Error sanitizing: {str(e)}>"

        return sanitized

    def _safe_sample(self, arr: np.ndarray, max_elements: int = 5) -> str:
        """Get a safe sample from array"""
        try:
            flat = arr.flatten()
            if len(flat) <= max_elements:
                return str(flat.tolist())
            else:
                sample = flat[:max_elements].tolist()
                return f"{sample} ... (total: {len(flat)} elements)"
        except Exception:
            return "<unable to sample>"

    def _sanitize_config(self, config: Dict) -> Dict:
        """Sanitize configuration data"""
        if not config:
            return {}

        sanitized = {}
        for k, v in config.items():
            if isinstance(v, dict):
                sanitized[k] = self._sanitize_config(v)
            elif isinstance(v, (list, tuple)) and len(v) > 20:
                sanitized[k] = f"{type(v).__name__}(length={len(v)})"
            else:
                sanitized[k] = v
        return sanitized

    def to_json(self, **kwargs) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(**kwargs), indent=2, default=str)

    def log(self, logger, as_json: bool = True):
        """
        Log error with appropriate level

        Args:
            logger: Logger instance
            as_json: If True, log as structured JSON; otherwise log as formatted text
        """
        if as_json:
            error_dict = self.to_dict()
            log_msg = json.dumps(error_dict, indent=2, default=str)
        else:
            log_msg = self._format_text()

        if self.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_msg)
        elif self.severity == ErrorSeverity.ERROR:
            logger.error(log_msg)
        elif self.severity == ErrorSeverity.WARNING:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

    def _format_text(self) -> str:
        """Format error as human-readable text"""
        lines = [
            "=" * 80,
            f"ERROR: {self.error_code or 'UNKNOWN'}",
            f"Severity: {self.severity.value.upper()}",
            f"Category: {self.category.value}",
            "=" * 80,
            f"Message: {self.message}",
        ]

        if self.component or self.operation:
            lines.append(f"Location: {self.component}.{self.operation}()")

        if self.model_name:
            lines.append(f"Model: {self.model_name}")

        if self.dataflow_names:
            lines.append(f"Dataflow: {', '.join(self.dataflow_names)}")

        if self.tensor_names:
            lines.append(f"Tensors: {', '.join(self.tensor_names)}")

        if self.exception_type:
            lines.append(f"Exception Type: {self.exception_type}")

        if self.input_data:
            sanitized_input = self._sanitize_data(self.input_data)
            lines.append(f"\nInput Data: {sanitized_input}")

        if self.expected_data:
            lines.append(f"\nExpected Data: {self.expected_data}")

        if self.actual_data:
            sanitized_actual = self._sanitize_data(self.actual_data)
            lines.append(f"\nActual Data: {sanitized_actual}")

        if self.related_config:
            lines.append(f"\nRelated Config: {self.related_config}")

        if self.troubleshooting_hints:
            lines.append("\nTroubleshooting Hints:")
            for i, hint in enumerate(self.troubleshooting_hints, 1):
                lines.append(f"  {i}. {hint}")

        if self.remediation_steps:
            lines.append("\nRemediation Steps:")
            for i, step in enumerate(self.remediation_steps, 1):
                lines.append(f"  {i}. {step}")

        if self.stack_trace:
            lines.append("\nStack Trace:")
            lines.append(self.stack_trace)

        if self.caused_by:
            lines.append("\nCaused by:")
            lines.append(f"  {self.caused_by}")

        lines.append("=" * 80)

        return "\n".join(lines)


class ErrorFactory:
    """
    Factory for creating standardized errors with consistent codes and hints

    This factory maintains an error catalog with predefined error codes, categories,
    and troubleshooting guidance. It ensures consistent error handling across the codebase.

    The factory can automatically extract component and operation names from context:
    - Pass `caller=self` to auto-extract class name as component
    - Pass `auto_operation=True` to auto-detect calling method name
    """

    # Error code definitions with troubleshooting hints
    ERROR_CATALOG: ClassVar[Dict[str, Dict[str, Any]]] = {
        # DataFlow errors (DF)
        "ERR_DF_001": {
            "message": "Required tensor not found in dataflow input",
            "category": ErrorCategory.DATA_VALIDATION,
            "severity": ErrorSeverity.ERROR,
            "hints": [
                "Check that all required input tensors are provided",
                "Verify the tensor names match the configuration",
                "Consider marking the tensor as optional if it's not always needed",
                "Review the model input configuration"
            ],
            "remediation": [
                "Add the missing tensor to the input data",
                "Update configuration to mark tensor as optional",
                "Check upstream processor output",
                "Verify route configuration"
            ]
        },
        "ERR_DF_002": {
            "message": "Invalid data type in custom data processing",
            "category": ErrorCategory.DATA_VALIDATION,
            "severity": ErrorSeverity.ERROR,
            "hints": [
                "Ensure base64 binary data is provided as bytes or string",
                "Check data encoding format",
                "Verify MIME type matches data format",
                "Review data type conversions"
            ],
            "remediation": [
                "Convert data to expected format (bytes or string)",
                "Verify base64 encoding is correct",
                "Check data type in request payload"
            ]
        },
        "ERR_DF_003": {
            "message": "Asset not found",
            "category": ErrorCategory.ASSET_NOT_FOUND,
            "severity": ErrorSeverity.ERROR,
            "hints": [
                "Verify asset ID exists in AssetManager",
                "Check if asset was properly uploaded/registered",
                "Ensure asset hasn't been deleted or expired",
                "Verify asset path and permissions"
            ],
            "remediation": [
                "Upload the asset before referencing it",
                "Check asset ID spelling/format",
                "Verify asset lifecycle management",
                "Check AssetManager logs for registration issues"
            ]
        },
        "ERR_DF_004": {
            "message": "Multiple result queues not supported in single dataflow",
            "category": ErrorCategory.CONFIGURATION,
            "severity": ErrorSeverity.ERROR,
            "hints": [
                "Restructure dataflow to have single queue output",
                "Split into multiple dataflows if needed"
            ],
            "remediation": [
                "Refactor to use single queue per dataflow",
                "Create separate dataflows for multiple queues"
            ]
        },
        "ERR_DF_005": {
            "message": "Queue full, dropping frames",
            "category": ErrorCategory.QUEUE,
            "severity": ErrorSeverity.WARNING,
            "hints": [
                "Consumer is too slow, increase processing speed",
                "Increase queue size if appropriate",
                "Check for bottlenecks in pipeline"
            ],
            "remediation": [
                "Optimize downstream processing",
                "Increase queue maxsize",
                "Add backpressure handling"
            ]
        },
        "ERR_DF_006": {
            "message": "Failed to inject tensors into input flows",
            "category": ErrorCategory.DATA_VALIDATION,
            "severity": ErrorSeverity.ERROR,
            "hints": [
                "No matching input tensors found in the request"
            ],
            "remediation": [
                "Provide at least one valid tensor in the request",
                "Check input tensor names against model configuration",
                "Verify request payload format",
                "Review input flow configuration"
            ]
        },

        # Model Operator errors (MO)
        "ERR_MO_001": {
            "message": "Empty result from preprocessing",
            "category": ErrorCategory.PROCESSING,
            "severity": ErrorSeverity.ERROR,
            "hints": [
                "Check preprocessor logic and return values",
                "Verify input data is valid",
                "Review preprocessor configuration",
                "Ensure preprocessor returns non-empty data"
            ],
            "remediation": [
                "Debug preprocessor implementation",
                "Verify input data format",
                "Check preprocessor output configuration",
                "Review preprocessing chain"
            ]
        },
        "ERR_MO_002": {
            "message": "Model output incomplete",
            "category": ErrorCategory.PROCESSING,
            "severity": ErrorSeverity.ERROR,
            "hints": [
                "Verify model generates all required outputs",
                "Check if postprocessor is needed",
                "Review model configuration",
                "Ensure all output tensors are generated"
            ],
            "remediation": [
                "Add missing output tensors",
                "Implement required postprocessor",
                "Update model output configuration",
                "Check model implementation"
            ]
        },
        "ERR_MO_003": {
            "message": "No output dataflow bound to model",
            "category": ErrorCategory.CONFIGURATION,
            "severity": ErrorSeverity.CRITICAL,
            "hints": [
                "Check route configuration",
                "Ensure model output is connected to downstream",
                "Review pipeline setup"
            ],
            "remediation": [
                "Add output binding in route configuration",
                "Connect model to output dataflow",
                "Review inference pipeline setup"
            ]
        },
        "ERR_MO_004": {
            "message": "Invalid result from preprocessing",
            "category": ErrorCategory.PROCESSING,
            "severity": ErrorSeverity.ERROR,
            "hints": [
                "Preprocessor must return valid data structure",
                "Check for errors in preprocessing chain",
                "Verify data format compatibility"
            ]
        },

        # Processor errors (PROC)
        "ERR_PROC_001": {
            "message": "Processor output count mismatch",
            "category": ErrorCategory.CONFIGURATION,
            "severity": ErrorSeverity.WARNING,
            "hints": [
                "Verify processor returns correct number of outputs",
                "Check processor configuration matches implementation",
                "Review processor output specification in the configuration",
                "Ensure output list length matches config"
            ],
            "remediation": [
                "Update processor to return correct number of outputs",
                "Fix processor configuration",
                "Review processor implementation"
            ]
        },
        "ERR_PROC_002": {
            "message": "Failed to load AutoProcessor",
            "category": ErrorCategory.CONFIGURATION,
            "severity": ErrorSeverity.CRITICAL,
            "hints": [
                "Verify model home path is correct",
                "Check if model supports AutoProcessor",
                "Ensure transformers library is installed",
                "Verify model files are present"
            ],
            "remediation": [
                "Check model_home path",
                "Download model files",
                "Use custom processor instead",
                "Install required dependencies"
            ]
        },
        "ERR_PROC_003": {
            "message": "Custom processor module invalid",
            "category": ErrorCategory.CONFIGURATION,
            "severity": ErrorSeverity.CRITICAL,
            "hints": [
                "Custom module must have create_instance() method",
                "Check custom processor implementation",
                "Verify module imports correctly"
            ],
            "remediation": [
                "Implement create_instance() in custom module",
                "Fix custom processor module",
                "Check module path and imports"
            ]
        },
        "ERR_PROC_004": {
            "message": "Failed to create custom processor",
            "category": ErrorCategory.CONFIGURATION,
            "severity": ErrorSeverity.CRITICAL,
            "hints": [
                "Check processor configuration",
                "Verify processor class exists",
                "Verify the processor name in the configuration matches the name in the processor class"
            ]
        },

        # Image processing errors (IMG)
        "ERR_IMG_001": {
            "message": "Unsupported image format",
            "category": ErrorCategory.DATA_VALIDATION,
            "severity": ErrorSeverity.ERROR,
            "hints": [
                "Supported formats: JPEG, PNG",
                "Check MIME type of uploaded image",
                "Convert image to supported format",
                "Verify image encoding"
            ],
            "remediation": [
                "Convert image to JPEG or PNG",
                "Check image MIME type",
                "Re-encode image in supported format"
            ]
        },
        "ERR_IMG_002": {
            "message": "Invalid base64 image data",
            "category": ErrorCategory.DATA_VALIDATION,
            "severity": ErrorSeverity.ERROR,
            "hints": [
                "Base64 image must be bytes or string",
                "Check data encoding",
                "Verify base64 format: data:image/[type];base64,[data]"
            ],
            "remediation": [
                "Encode image as valid base64 string",
                "Check data URL format",
                "Verify data type (bytes/string)"
            ]
        },
        "ERR_IMG_003": {
            "message": "Image decoder not initialized",
            "category": ErrorCategory.INTERNAL,
            "severity": ErrorSeverity.CRITICAL,
            "hints": [
                "Image decoder failed to initialize",
                "Check codec dependencies",
                "Verify system libraries"
            ]
        },

        # Video processing errors (VIDEO)
        "ERR_VIDEO_001": {
            "message": "Invalid video frame sampling parameters",
            "category": ErrorCategory.CONFIGURATION,
            "severity": ErrorSeverity.WARNING,
            "hints": [
                "Frames or chunks parameters differ across assets",
                "Ensure consistent parameters for batch processing",
                "Review asset query parameters"
            ]
        },

        # Route configuration errors (ROUTE)
        "ERR_ROUTE_001": {
            "message": "Invalid route configuration",
            "category": ErrorCategory.CONFIGURATION,
            "severity": ErrorSeverity.CRITICAL,
            "hints": [
                "Check route format: 'model:[tensor, ...]' -> 'model:[tensor, ...]'",
                "Verify JSON syntax in data specifications",
                "Review route configuration documentation"
            ],
            "remediation": [
                "Fix route configuration format",
                "Validate JSON in route data",
                "Review routing examples"
            ]
        },
        "ERR_ROUTE_002": {
            "message": "Model not found in route configuration",
            "category": ErrorCategory.CONFIGURATION,
            "severity": ErrorSeverity.CRITICAL,
            "hints": [
                "Model referenced in route doesn't exist",
                "Check model name spelling",
                "Verify model is defined in configuration"
            ],
            "remediation": [
                "Add model to configuration",
                "Fix model name in route",
                "Review model definitions"
            ]
        },

        # AsyncDispatcher errors (ASYNC)
        "ERR_ASYNC_001": {
            "message": "AsyncDispatcher failed to start - collector is None",
            "category": ErrorCategory.INTERNAL,
            "severity": ErrorSeverity.CRITICAL,
            "hints": [
                "Collector not properly initialized",
                "Check pipeline setup",
                "Review initialization order"
            ]
        },
        "ERR_ASYNC_002": {
            "message": "AsyncDispatcher failed to start - event loop is None",
            "category": ErrorCategory.INTERNAL,
            "severity": ErrorSeverity.CRITICAL,
            "hints": [
                "Event loop not properly initialized",
                "Check async setup",
                "Review initialization order"
            ]
        },
        "ERR_ASYNC_003": {
            "message": "Consumer queue reached maximum size",
            "category": ErrorCategory.QUEUE,
            "severity": ErrorSeverity.CRITICAL,
            "hints": [
                "Too many concurrent requests",
                "Increase queue size",
                "Add rate limiting",
                "Check consumer processing speed"
            ],
            "remediation": [
                "Increase n_max_async_queues parameter",
                "Implement request throttling",
                "Optimize consumer processing"
            ]
        },

        # System errors (SYS)
        "ERR_SYS_001": {
            "message": "Model repository does not exist",
            "category": ErrorCategory.CONFIGURATION,
            "severity": ErrorSeverity.CRITICAL,
            "hints": [
                "Check model_repo path in configuration",
                "Verify directory exists and is accessible",
                "Check file permissions"
            ],
            "remediation": [
                "Create the model repository directory and correctly mount it into the container.",
                "Fix model_repo path in configuration",
                "Check directory permissions"
            ]
        },
        "ERR_SYS_002": {
            "message": "Inference pipeline incomplete",
            "category": ErrorCategory.CONFIGURATION,
            "severity": ErrorSeverity.CRITICAL,
            "hints": [
                "Either input or output is empty",
                "Check route configuration",
                "Review pipeline setup"
            ],
            "remediation": [
                "Add input/output bindings",
                "Fix route configuration",
                "Review pipeline configuration"
            ]
        },
    }

    @classmethod
    def _get_caller_info(cls, caller=None, auto_operation: bool = False):
        """
        Extract component and operation names from caller context

        Args:
            caller: The calling object (usually `self`). If provided, extracts class name.
            auto_operation: If True, extracts the calling method name from stack

        Returns:
            tuple: (component_name, operation_name)
        """
        component = None
        operation = None

        # Extract component from caller object
        if caller is not None:
            component = caller.__class__.__name__

        # Extract operation from call stack
        if auto_operation:
            # Get the calling frame (skip this method and the create/from_exception method)
            frame = inspect.currentframe()
            try:
                # Go up the stack: _get_caller_info -> create/from_exception -> actual caller
                caller_frame = frame.f_back.f_back
                if caller_frame:
                    operation = caller_frame.f_code.co_name
            finally:
                del frame  # Avoid reference cycles

        return component, operation

    @classmethod
    def create(
        cls,
        error_code: str,
        message: Optional[str] = None,
        caller: Optional[Any] = None,
        auto_operation: bool = True,
        **kwargs
    ) -> EnhancedError:
        """
        Create error from catalog with additional context

        Args:
            error_code: Error code from ERROR_CATALOG
            message: Custom message (overrides catalog message)
            caller: The calling object (usually `self`). Auto-extracts class name as component.
            auto_operation: If True, auto-detects calling method name as operation.
            **kwargs: Additional context fields for EnhancedError

        Returns:
            EnhancedError instance with all context

        Example:
            # Automatic component/operation detection:
            error = ErrorFactory.create(
                "ERR_DF_001",
                caller=self,              # Extracts "DataFlow" from self.__class__.__name__
                auto_operation=True,      # Extracts "put" from calling method
                tensor_names=["image"]
            )

            # Manual specification (still supported):
            error = ErrorFactory.create(
                "ERR_DF_001",
                component="DataFlow",
                operation="put",
                tensor_names=["image"]
            )
        """
        # Auto-detect component and operation if requested
        auto_component, auto_operation_name = cls._get_caller_info(caller, auto_operation)

        # Use auto-detected values if not explicitly provided
        if auto_component and 'component' not in kwargs:
            kwargs['component'] = auto_component
        if auto_operation_name and 'operation' not in kwargs:
            kwargs['operation'] = auto_operation_name

        if error_code not in cls.ERROR_CATALOG:
            # Remove conflicting keys from kwargs
            final_category = kwargs.pop('category', ErrorCategory.INTERNAL)
            final_severity = kwargs.pop('severity', ErrorSeverity.ERROR)
            error = EnhancedError(
                message=message or f"Unknown error code: {error_code}",
                category=final_category,
                severity=final_severity,
                error_code=error_code,
                troubleshooting_hints=[
                    "This error code is not in the catalog",
                    "Check error code spelling",
                    "Review error handling implementation"
                ],
                **kwargs
            )
            # Add to global collector if enabled
            cls._add_to_collector(error)
            return error

        template = cls.ERROR_CATALOG[error_code]

        # Use custom message or template message
        final_message = message or template["message"]

        # If custom message provided, append template message as hint
        hints = list(template.get("hints", []))
        if message and message != template["message"]:
            hints.insert(0, f"Base error: {template['message']}")

        # Merge catalog values with provided kwargs
        # User-provided kwargs override catalog defaults
        final_category = kwargs.pop('category', template.get("category", ErrorCategory.INTERNAL))
        final_severity = kwargs.pop('severity', template.get("severity", ErrorSeverity.ERROR))

        error = EnhancedError(
            message=final_message,
            category=final_category,
            severity=final_severity,
            error_code=error_code,
            troubleshooting_hints=hints,
            remediation_steps=list(template.get("remediation", [])),
            **kwargs
        )
        # Add to global collector if enabled
        cls._add_to_collector(error)
        return error

    @classmethod
    def _add_to_collector(cls, error: EnhancedError):
        """Add error to global collector if enabled"""
        global _global_error_collector
        if _global_error_collector is not None:
            _global_error_collector.add(error)

    @classmethod
    def from_exception(
        cls,
        exc: Exception,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        caller: Optional[Any] = None,
        auto_operation: bool = True,
        error_code: Optional[str] = None,
        category: ErrorCategory = ErrorCategory.INTERNAL,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        **kwargs
    ) -> EnhancedError:
        """
        Create EnhancedError from Python exception

        Args:
            exc: The exception object
            component: Component where error occurred (or use caller for auto-detection)
            operation: Operation that failed (or use auto_operation=True)
            caller: The calling object (usually `self`). Auto-extracts class name as component.
            auto_operation: If True, auto-detects calling method name as operation.
            error_code: Optional error code
            category: Error category
            severity: Error severity
            **kwargs: Additional context

        Returns:
            EnhancedError with exception details

        Example:
            # Automatic component/operation detection:
            except Exception as e:
                error = ErrorFactory.from_exception(
                    e,
                    caller=self,           # Extracts class name
                    auto_operation=True    # Extracts method name
                )

            # Manual specification (still supported):
            except Exception as e:
                error = ErrorFactory.from_exception(
                    e,
                    component="ModelOperator",
                    operation="run"
                )
        """
        # Auto-detect component and operation if requested
        auto_component, auto_operation_name = cls._get_caller_info(caller, auto_operation)

        # Use auto-detected values if not explicitly provided
        if component is None and auto_component:
            component = auto_component
        if operation is None and auto_operation_name:
            operation = auto_operation_name

        # Remove conflicting keys from kwargs to avoid duplicate parameter errors
        # (kwargs might contain category/severity if passed by caller)
        kwargs.pop('category', None)
        kwargs.pop('severity', None)
        kwargs.pop('component', None)
        kwargs.pop('operation', None)
        kwargs.pop('error_code', None)

        error = EnhancedError(
            message=str(exc),
            category=category,
            severity=severity,
            component=component,
            operation=operation,
            error_code=error_code or "ERR_EXCEPTION",
            exception_type=type(exc).__name__,
            stack_trace=traceback.format_exc(),
            troubleshooting_hints=[
                f"Exception type: {type(exc).__name__}",
                f"Exception message: {str(exc)}",
                "Review stack trace for details"
            ],
            **kwargs
        )
        # Add to global collector if enabled
        cls._add_to_collector(error)
        return error

    @classmethod
    def get_catalog_summary(cls) -> Dict[str, Any]:
        """Get summary of error catalog for documentation"""
        summary = {
            "total_errors": len(cls.ERROR_CATALOG),
            "by_category": {},
            "by_severity": {},
            "errors": {}
        }

        for code, info in cls.ERROR_CATALOG.items():
            category = info.get("category", ErrorCategory.INTERNAL).value
            severity = info.get("severity", ErrorSeverity.ERROR).value

            summary["by_category"][category] = summary["by_category"].get(category, 0) + 1
            summary["by_severity"][severity] = summary["by_severity"].get(severity, 0) + 1

            summary["errors"][code] = {
                "message": info["message"],
                "category": category,
                "severity": severity,
                "hints_count": len(info.get("hints", [])),
                "remediation_count": len(info.get("remediation", []))
            }

        return summary


class ErrorCollector:
    """
    Collect and analyze errors for patterns and statistics

    This class maintains a thread-safe collection of errors and provides
    statistics and analysis capabilities for monitoring and debugging.
    """

    def __init__(self, max_errors: int = 1000):
        """
        Initialize error collector

        Args:
            max_errors: Maximum number of errors to keep in memory
        """
        self._errors: List[EnhancedError] = []
        self._error_counts: Dict[str, int] = {}
        self._max_errors = max_errors
        self._lock = threading.RLock()  # RLock allows same thread to acquire multiple times
        self._start_time = time.time()

    def add(self, error: EnhancedError):
        """Add error to collection (thread-safe)"""
        with self._lock:
            self._errors.append(error)
            self._error_counts[error.error_code] = \
                self._error_counts.get(error.error_code, 0) + 1

            # Trim if exceeds max size (keep most recent)
            if len(self._errors) > self._max_errors:
                self._errors = self._errors[-self._max_errors:]

    def get_stats(self, include_recent: int = 10) -> Dict:
        """
        Get error statistics for monitoring

        Args:
            include_recent: Number of recent errors to include

        Returns:
            Dictionary with error statistics
        """
        with self._lock:
            stats = {
                "collection_period_seconds": time.time() - self._start_time,
                "total_errors": len(self._errors),
                "unique_error_codes": len(self._error_counts),
                "by_category": self._count_by("category"),
                "by_severity": self._count_by("severity"),
                "by_component": self._count_by("component"),
                "by_model": self._count_by("model_name"),
                "by_code": dict(sorted(
                    self._error_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:20]),  # Top 20 error codes
            }

            if include_recent > 0:
                stats["recent_errors"] = [
                    e.to_dict(include_stack_trace=False)
                    for e in self._errors[-include_recent:]
                ]

            return stats

    def get_errors_by_category(self, category: ErrorCategory) -> List[EnhancedError]:
        """Get all errors of a specific category"""
        with self._lock:
            return [e for e in self._errors if e.category == category]

    def get_errors_by_code(self, error_code: str) -> List[EnhancedError]:
        """Get all errors with specific error code"""
        with self._lock:
            return [e for e in self._errors if e.error_code == error_code]

    def get_critical_errors(self) -> List[EnhancedError]:
        """Get all critical errors"""
        with self._lock:
            return [e for e in self._errors if e.severity == ErrorSeverity.CRITICAL]

    def clear(self):
        """Clear all collected errors"""
        with self._lock:
            self._errors.clear()
            self._error_counts.clear()
            self._start_time = time.time()

    def _count_by(self, field: str) -> Dict[str, int]:
        """Count errors by field value"""
        counts = {}
        for error in self._errors:
            value = getattr(error, field, None)
            if value is None or value == "":
                continue

            # Handle enum values
            if hasattr(value, 'value'):
                key = value.value
            # Handle lists
            elif isinstance(value, list):
                if len(value) == 0:
                    continue
                key = ", ".join(str(v) for v in value[:3])  # First 3 items
                if len(value) > 3:
                    key += "..."
            else:
                key = str(value)

            counts[key] = counts.get(key, 0) + 1

        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:20])

    def export_to_json(self, filepath: str, include_stack_traces: bool = False):
        """Export all errors to JSON file"""
        with self._lock:
            data = {
                "exported_at": time.time(),
                "stats": self.get_stats(include_recent=0),
                "errors": [
                    e.to_dict(include_stack_trace=include_stack_traces)
                    for e in self._errors
                ]
            }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())


# Global error collector instance (optional)
_global_error_collector = None

def get_global_error_collector() -> ErrorCollector:
    """Get or create global error collector"""
    global _global_error_collector
    if _global_error_collector is None:
        _global_error_collector = ErrorCollector()
    return _global_error_collector


def enable_global_error_collection(max_errors: int = 1000):
    """Enable global error collection"""
    global _global_error_collector
    _global_error_collector = ErrorCollector(max_errors=max_errors)


def disable_global_error_collection():
    """Disable global error collection"""
    global _global_error_collector
    _global_error_collector = None

