# SPDX-License-Identifier: MIT
# Copyright (c) 2011–2026 Joris J.C. Remmers

import copy
from typing import Any, Union
import numpy as np


class BaseMaterial:
    """
    Base class for material models in finite element analysis.

    This class provides the fundamental structure and methods for implementing
    constitutive material models. It handles material properties and history
    variables for path-dependent materials.

    Attributes
    ----------
    oldHistory : Dict[str, Any]
        Dictionary storing history variables from the previous converged step.
    newHistory : Dict[str, Any]
        Dictionary storing history variables for the current step.
    """

    def __init__(self) -> None:
        """
        Initialize the BaseMaterial instance.
        """

        self.oldHistory = {}
        self.newHistory = {}

    def hasHistory(self, elementID, intpointID, name) -> bool:
        """
        Check whether a history parameter exists for a specific material point.

        Parameters
        ----------
        elementID : int
            Element number
        intpointID : int
            Integration point number
        name : str
            Name of the history parameter.

        Returns
        -------
        bool
            True if the history parameter exists, False otherwise.
        """
        return (elementID, intpointID, name) in self.oldHistory

    def setHistoryParameter(self, elementID, intpointID, name: str, val: Any) -> None:
        """
        Set a history parameter for the current step.

        History parameters are used to store internal state variables for
        path-dependent material models (e.g., plastic strain, damage variables).

        Parameters
        ----------
        elementID : int
            Element number
        intpointID : int
            Integration point number
        name : str
            Name of the history parameter.
        val : Any
            Value of the history parameter (typically float or ndarray).
        """
        self.newHistory[(elementID, intpointID, name)] = val
        return

    def getHistoryParameter(
        self, elementID, intpointID, name: str
    ) -> Union[float, np.ndarray]:
        """
        Retrieve a history parameter from the previous converged step.

        Parameters
        ----------
        elementID : int
            Element number
        intpointID : int
            Integration point number
        name : str
            Name of the history parameter to retrieve.

        Returns
        -------
        Union[float, ndarray]
            The value of the history parameter. Returns a copy for array types
            to prevent unintended modifications.
        """
        val = self.oldHistory[(elementID, intpointID, name)]

        if type(val) == float:
            return val
        else:
            return val.copy()

    def commit(self) -> None:
        """
        Commit the current history variables to old history.

        This method is called when a load step has converged, copying the
        current (new) history variables to become the old history variables
        for the next step. Uses deep copy to ensure complete independence.
        """
        self.oldHistory = copy.deepcopy(self.newHistory)
