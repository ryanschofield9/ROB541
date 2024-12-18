#! /usr/bin/python3
from collections import UserList
import numpy as np
from operator import methodcaller
from . import utilityfunctions as ut
from . import core


class Manifold:
    """
    Class to hold manifold structure
    """

    def __init__(self,
                 transition_table,
                 n_dim):
        # Save the provided chart transition table as a class instance attribute
        self.transition_table = transition_table
        # Extract the number of charts implied by the transition table
        self.n_charts = len(transition_table)
        # Save the provided dimensionality as a class instance attribute
        self.n_dim = n_dim

    def element(self,
                value,
                initial_chart=0):
        """Instantiate a manifold element with a specified value"""
        q = ManifoldElement(self,
                            value,
                            initial_chart)
        return q

    def element_set(self,
                    value=None,
                    initial_chart=0,
                    input_format=None):
        q_set = ManifoldElementSet(self,
                                   value,
                                   initial_chart,
                                   input_format)

        return q_set

    @property
    def element_shape(self):
        return (self.n_dim,)


class ManifoldElement(core.GeomotionElement):
    """
    Class for manifold elements
    """

    def __init__(self,
                 manifold,
                 value=None,
                 initial_chart=0):

        # Check if the first argument is a ManifoldElement already, and if so, extract its value and manifold
        if isinstance(manifold, ManifoldElement):
            manifold_element_input = manifold
            self.value = manifold_element_input.value
            self.manifold = manifold_element_input.manifold
            self.current_chart = manifold_element_input.current_chart

        # If the first argument is not a manifold element, use the full set of provided inputs
        else:
            # Save the provided manifold, configuration, and initial chart as class instance attributes
            self.manifold = manifold

            # Make sure that the value is an ndarray
            value = ut.ensure_ndarray(value)

            self.current_chart = initial_chart
            self.value = value

        # Information about how to build a set of these objects
        self.plural = ManifoldElementSet

    def format_value(self, val):

        # Make sure the value is an ndarray
        val = ut.ensure_ndarray(val)

        # Check element shape
        if val.shape == self.manifold.element_shape:
            pass
        else:
            raise Exception("Provided value does not match element shape for manifold")

        # return the formatted and verified value
        return val

    def transition(self, new_chart):
        """
        Change the chart used to describe the manifold element, updating the numbers describing
        the configuration so that the actual point on the manifold stays the same
        """

        # Simple passthrough behavior if chart is not actually changing
        if new_chart == self.current_chart:

            new_value = self.value

        # Raise an exception if the transition from the current to new chart is not defined
        elif self.manifold.transition_table[self.current_chart][new_chart] is None:

            raise Exception(
                "The transition from " + str(self.current_chart) + " to " + str(new_chart) + " is undefined.")

        # If transition is non-trivial and defined, use the specified transition function
        else:

            new_value = self.manifold.transition_table[self.current_chart][new_chart](self.value)

        # Return an object of the same class with the new value and new chart
        # (calling self.__class__ ensures compatibility with subclasses of ManifoldElement)
        return self.__class__(self.manifold, new_value, new_chart)


class ManifoldElementSet(core.GeomotionSet):
    """ Argument list should either be a list of manifold elements or
    Manifold, GridArray, initial_chart, component-or-element """

    def __init__(self,
                 manifold,
                 value=None,
                 initial_chart=0,
                 input_format=None):
        """ First input is one of:
        1. A ManifoldElementSet (which gets passed through into a copy of the original)
        2. A ManifoldElement (which gets wrapped into a single-element list)
        3. A nested list-of-lists of ManifoldElements
        4. a Manifold, followed by
           a GridArray of values
           an initial chart (either a single value or an element-outer grid
           (optional) component-outer or element-outer specification for grid"""

        # Check if the first argument is a ManifoldElementSet already, and if so, extract its value and manifold
        if isinstance(manifold, ManifoldElementSet):
            manifold_element_set_input = manifold
            value = manifold_element_set_input.value
            manifold = manifold_element_set_input.manifold

        # Check if the first argument is a bare manifold element, and if so, wrap its value in a list
        elif isinstance(manifold, ManifoldElement):
            manifold_element_input = manifold
            value = ut.ensure_list(manifold_element_input)
            manifold = manifold_element_input.manifold

        # Check if the first argument is a list-of-lists of Elements of the specified type, and if so, use it directly
        elif isinstance(manifold, list):
            list_input = manifold
            if ut.object_list_all_instance(ManifoldElement, list_input):
                value = list_input
                manifold = ut.object_list_extract_first_entry(list_input).manifold
            else:
                raise Exception("List input to ManifoldElementSet should contain ManifoldElement objects",
                                "not ", type(ut.object_list_extract_first_entry(list_input)))

        # If the first argument is a manifold, process the inputs as if they were ManifoldElement inputs
        # with values provided in a GridArray
        elif isinstance(manifold, Manifold):
            if isinstance(value, ut.GridArray):

                # Extract the grid array from the argument list
                grid = value

                # Make sure that the grid is in element-outer format
                grid = ut.format_grid(grid, manifold.element_shape, 'element', input_format)

                # Handle possibility for manifold elements to have individual chart specifications or
                # one shared specification

                # Check for initial_chart being a GridArray
                if isinstance(initial_chart, ut.GridArray):
                    # Make sure it matches the dimensions of the value grid
                    if initial_chart.shape == grid.shape[:grid.n_outer]:
                        # Construct a manifold element with each value/chart pair in the grids
                        value = ut.object_list_eval_pairwise(manifold.element,
                                                             grid, initial_chart, grid.n_outer)
                    else:
                        raise Exception("Initial_chart is a grid that doesn't match the value grids")
                else:
                    # Preload the initial chart into the manifold element constructor, and evaluate
                    # it for each configuration
                    def manifold_element_construction_function(manifold_element_value):
                        return manifold.element(manifold_element_value, initial_chart)

                    value = ut.object_list_eval(manifold_element_construction_function, grid, grid.n_outer)

            else:
                raise Exception(
                    "If first input to ManifoldElementSet is a Manifold, second input should be a GridArray")

        else:
            raise Exception("First argument to ManifoldSet should be either a list-of-lists of "
                            "Elements or a Manifold")

        super().__init__(value)
        self.manifold = manifold

        # Information about what objects this set should contain
        self.single = ManifoldElement

    @property
    def element_shape(self):
        return (self.manifold.n_dim,)

    @property
    def grid(self):
        def extract_value(x):
            return x.value

        # Get an array of the manifold element values, and use nested_stack to make it an ndarray
        element_outer_grid = ut.nested_stack(ut.object_list_eval(extract_value,
                                                                 self.value))

        # Convert this array into a GridArray
        element_outer_grid_array = ut.GridArray(element_outer_grid, n_inner=len(self.element_shape))

        component_outer_grid_array = element_outer_grid_array.everse

        return component_outer_grid_array

    def transition(self, new_chart):

        if isinstance(new_chart, (int, float)):
            transition_method = methodcaller('transition', new_chart)
            new_set = ut.object_list_eval(transition_method,
                                          self.value)
        elif isinstance(new_chart, ut.GridArray):
            new_set = ut.object_list_method_eval_pairwise('transition', self.value, new_chart)

        else:
            raise Exception("New chart should be specified as an int or a grid array")

        return self.__class__(new_set)


class ManifoldFunction:
    """ManifoldFunction acts as a wrapper around a numerical function, such that it takes input as a ManifoldElement
    or ManifoldElementSet and automatically converts it into the chart in which the function is defined,
    then into numerical data for the underlying function to act on. There is also the option to provide
    post-processing functions for elements and sets; this functionality is used by child classes such as
    ManifoldMap and TangentVectorField"""

    def __init__(self,
                 manifold: Manifold,
                 defining_function_list,  # Underlying numeric function
                 postprocess_function=None):  # How to format the output of the numeric function

        """defining function_list is a list of functions defining the function on the corresponding charts.
        Where the charts overlap, the functions should agree with each other. If the function is not explicitly defined
        on a chart, the corresponding element of defining_function_list should be None, and an attempt will be made to
        construct the function on the chart via a pullback operation."""

        if not isinstance(defining_function_list, list):
            defining_function_list = ut.ensure_list(defining_function_list)

        if not len(defining_function_list) == manifold.n_charts:
            raise Exception("Defining function list does not match the number of charts on the manifold")

        # Attempt to pull back function onto charts for which it was not provided, by finding the lowest-numbered chart
        # with a defined transition map, and pulling back through that map. If no single-transition path is found,
        # provide a function that returns NaN
        for i, d in enumerate(defining_function_list):

            # If the function is not defined on the chart
            if d is None:

                # reset the counter for iterating through the charts
                j = 0
                transition_found = False

                # Check to see if a transition has been found
                while not transition_found:

                    # If no transition has been found, and there is a transition to chart
                    # j in which the function is defined
                    if (manifold.transition_table[i][j] is not None) and (defining_function_list[j] is not None):

                        # Mark the transition as having been found
                        transition_found = True

                        # Pull back the function through the transition map
                        defining_function_list[i] = core.PullbackFunction(defining_function_list[j],
                                                                          manifold.transition_table[i][j])

                    # If we've checked all the transitions and did not find one that
                    # takes us to a chart in which the function is defined, then define a NaN-valued function
                    elif j == manifold.n_charts - 1:
                        defining_function_list[i] = lambda x: np.NaN
                        transition_found = True

                    # If we haven't checked all the transitions, increment the count and check the next one
                    else:
                        j = j + 1


        # Save the inputs as instance properties
        self.manifold = manifold
        self.defining_function_list = defining_function_list
        self.postprocess_function = postprocess_function

    def __call__(self, configuration, *args, **kwargs):
        """Break down the provided configuration element(s) to numeric values in the defining chart, then
        apply the underlying function and any output processing"""

        # Convert the configuration value into an element-wise GridArray of numeric data
        configuration_grid_e, function_index_list, value_type = self.preprocess(configuration)

        # Evaluate the defining function over the grid with any additional arguments provided
        function_grid_e = self.process(configuration_grid_e, function_index_list, *args, **kwargs)

        # Apply any post-process formatting
        function_value = self.postprocess(configuration_grid_e, function_grid_e, function_index_list, value_type)

        return function_value

    def preprocess(self, configuration):
        """Configuration_value input can be one of:
                    1. ManifoldElement
                    3. ManifoldElementSet"""

        # Record the single- or multiple-input status for post-processing
        if isinstance(configuration, ManifoldElement):
            value_type = 'single'
        elif isinstance(configuration, ManifoldElementSet):
            value_type = 'multiple'
        else:
            raise Exception("ManifoldFunction must be called with a ManifoldElement or ManifoldElementSet")

        # Put the input into ManifoldElementSet form (no change is made if it is already a set)
        configuration_set = ManifoldElementSet(configuration)

        # Extract a component-wise grid from the ManifoldElementSet and evert it to element-wise formatting
        configuration_grid_e = configuration_set.grid.everse

        # Extract the current chart from each element
        function_index_list = ut.object_list_eval(lambda x: x.current_chart, configuration_set)

        # Make function_index_list a grid_array
        function_index_list = ut.GridArray([function_index_list], n_outer=1).everse

        return configuration_grid_e, function_index_list, value_type

    def process(self, configuration_grid_e, function_index_list, *process_args, **kwargs):
        """Preload any non-configuration inputs that have been provided to the function, evaluate over the provided
        configurations, and return an element-wise grid of numeric values"""

        def defining_function_with_inputs(config, function_index):
            return self.defining_function_list[function_index[0]](config, *process_args, **kwargs)

        # Evaluate the defining function over the grid in the chart for which each point is defined
        function_grid_e = ut.GridArray(ut.array_eval_pairwise(defining_function_with_inputs, configuration_grid_e,
                                                              function_index_list, configuration_grid_e.n_outer),
                                       configuration_grid_e.n_outer)

        return function_grid_e

    def postprocess(self, configuration_grid_e, function_grid_e, function_index_list, value_type):
        """If the input was a single element, make the output also single element, and then apply the
        post-processing operation"""

        if self.postprocess_function is not None:
            if value_type == 'single':
                configuration_value = configuration_grid_e[0]  # Extract the single item from the config grid array
                function_value = function_grid_e[0]  # Extract the single item from the function grid array
                function_index = function_index_list[0]
                return self.postprocess_function[0](configuration_value, function_value, function_index)
            elif value_type == 'multiple':
                return self.postprocess_function[1](configuration_grid_e, function_grid_e, function_index_list)
            else:
                raise Exception("Value_type should be 'single' or 'multiple'.")
        else:
            # Default is to return a component-wise grid of the function value
            return function_grid_e.everse

    def pullback(self, pullback_function, *args, **kwargs):
        """Method that is equivalent to using self as the "outer" input to PullbackFunction"""

        return core.PullbackFunction(self, pullback_function, *args, **kwargs)


class ManifoldMap(ManifoldFunction):
    """A manifold function that post-processes the output from the numeric function into manifold elements"""

    def __init__(self,
                 manifold: Manifold,  # The manifold that input elements are part of
                 output_manifold: Manifold,  # The manifold that output elements are part of
                 defining_function_list,  # The underlying numeric function
                 output_defining_chart=None,  # The output-manifold chart in which the function range is defined
                 output_chart=None):  # An output chart to use, if different from the definition chart

        if not isinstance(defining_function_list, list):
            defining_function_list = [defining_function_list]

        # if defining_chart is None:
        #     defining_chart = [0] * len(defining_function_list)
        # elif not isinstance(defining_chart, list):
        #     defining_chart = [defining_chart]

        if output_defining_chart is None:
            output_defining_chart = [0] * len(defining_function_list)
        elif not isinstance(output_defining_chart, list):
            output_defining_chart = ut.ensure_list(output_defining_chart)

        # If a separate output chart is not specified, match it to the output defining chart
        if output_chart is None:
            output_chart = output_defining_chart
        else:
            if not isinstance(output_chart, list):
                output_chart = [output_chart]

        if not ut.shape(defining_function_list) == ut.shape(output_defining_chart):
            raise Exception("Defining function list and output defining chart list do not have matching shapes")
        elif not ut.shape(defining_function_list) == ut.shape(output_chart):
            raise Exception("Defining function list and output chart list do not have matching shapes")

        # Turn the numerical output of the defining function into manifold elements or
        # element sets in the output manifold
        def postprocess_function_single(q_input, q_output, function_index):
            return output_manifold.element(q_output,
                                           output_defining_chart[function_index[0]]). \
                transition(output_chart[function_index[0]])

        def postprocess_function_multiple(q_input, q_output, function_index_list):

            def get_output_defining_chart(function_index):
                return output_defining_chart[function_index[0]]

            def get_output_chart(function_index):
                return output_chart[function_index[0]]

            output_defining_chart_grid = function_index_list.grid_eval(get_output_defining_chart)

            output_chart_grid = function_index_list.grid_eval(get_output_chart)

            return self.output_manifold.element_set(q_output, output_defining_chart_grid).transition(output_chart_grid)

        postprocess_function = [postprocess_function_single, postprocess_function_multiple]

        # Initialize the standard pieces of a ManifoldFunction
        super().__init__(manifold,
                         defining_function_list,
                         postprocess_function)

        # Store the extra information associated with having manifold output
        self.output_defining_chart = output_defining_chart
        self.output_chart = output_chart
        self.output_manifold = output_manifold

    def transition_output(self, new_output_chart):
        return self.__class__(self.manifold,
                              self.output_manifold,
                              self.defining_function_list,
                              self.defining_chart,
                              self.output_defining_chart,
                              new_output_chart)
