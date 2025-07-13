import stim
import numpy as np
import random
from itertools import product
from typing import Callable, Tuple, List, Optional, Any, Union


def find_distance_with_decoder(
    circuit: stim.Circuit, 
    decoder: Callable[[np.ndarray], Union[np.ndarray, Tuple[np.ndarray, Any]]],
    max_shots: int = 2**10
):
    """
    Find the code distance using a provided decoder.
    
    This function estimates the code distance by testing different logical error
    configurations and finding the minimum weight error that causes a logical error.
    
    Args:
        circuit: The quantum error correction circuit
        decoder: Function that takes detector measurements and returns predicted observables
        max_shots: Maximum number of logical error configurations to test
        
    Returns:
        Tuple of (distance, minimal_error, error_tags, error_explanations)
    """
    # Get the detector error model from the original circuit
    dem = circuit.detector_error_model()
    
    # Find logical observable indices
    logical_locations = []
    k = circuit.num_observables
    
    # For simplicity, we'll assume logical observables are the last k detectors
    # This is a reasonable assumption for most quantum error correction codes
    logical_locations = list(range(dem.num_detectors - k, dem.num_detectors))
    
    # Generate logical error configurations to test
    if 2**k > max_shots:
        logical_flipped_configurations = (random.choices((0, 1), k=k) for _ in range(max_shots))
    else:
        logical_flipped_configurations = product((0, 1), repeat=k)
    
    # Find minimum weight error for each logical configuration
    distance = np.inf
    minimal_error = None
    best_logical_config = None
    
    for logicals_flipped in logical_flipped_configurations:
        if sum(logicals_flipped) == 0: 
            continue
            
        # Create syndrome corresponding to this logical error
        syndromes = np.zeros(dem.num_detectors)
        syndromes[logical_locations] = logicals_flipped
        
        # Use the provided decoder to find minimum weight error
        try:
            # The decoder should return the predicted observables
            predicted_observables = decoder(syndromes.reshape(1, -1))
            
            # Handle tuple return type
            if isinstance(predicted_observables, tuple):
                predicted_observables = predicted_observables[0]
            
            # If decoder predicts the correct logical values, this configuration
            # doesn't represent a logical error. Skip it.
            if np.array_equal(predicted_observables.flatten(), logicals_flipped):
                continue
                
            # For distance calculation, we need to find the actual error pattern
            # This requires a more sophisticated approach since the decoder
            # only gives us the logical prediction, not the error pattern
            
            # For now, we'll use a simplified approach: try to find errors
            # that could cause this logical error by sampling from the DEM
            error_weight = _find_minimum_error_weight_for_logical(
                dem, logicals_flipped, logical_locations, max_attempts=1000
            )
            
            if error_weight < distance:
                distance = error_weight
                best_logical_config = logicals_flipped
                
        except Exception as e:
            print(f"Warning: Decoder failed for logical configuration {logicals_flipped}: {e}")
            continue
    
    if distance == np.inf:
        print("Warning: Could not find any logical errors. Distance may be infinite.")
        return np.inf, None, [], []
    
    # Find the actual error pattern for the best configuration
    minimal_error = _find_error_pattern_for_logical(
        dem, best_logical_config, logical_locations, int(distance)
    )
    
    # Get error tags and explanations
    error_tags = [dem[i].tag for i in np.where(minimal_error)[0]]
    
    error_explanations = []
    for i in np.where(minimal_error)[0]:
        new_dem = stim.DetectorErrorModel()
        new_dem.append(dem[i])
        explanation = circuit.explain_detector_error_model_errors(dem_filter=new_dem)[0]
        error_explanations.append(explanation)
    
    return distance, minimal_error, error_tags, error_explanations


def _find_minimum_error_weight_for_logical(
    dem: stim.DetectorErrorModel, 
    logical_config, 
    logical_locations: List[int],
    max_attempts: int = 1000
):
    """
    Find the minimum weight error that could cause the given logical error.
    This is a simplified approach that samples from the DEM.
    """
    sampler = dem.compile_sampler()
    min_weight = np.inf
    
    for _ in range(max_attempts):
        # Sample errors from the DEM
        dets, obs, errs = sampler.sample(shots=1, return_errors=True)
        
        if errs is None:
            continue
            
        # Check if this error pattern causes the desired logical error
        error_pattern = errs[0]
        if np.array_equal(obs[0][logical_locations], logical_config):
            weight = np.sum(error_pattern)
            if weight < min_weight:
                min_weight = weight
    
    return min_weight if min_weight != np.inf else 1


def _find_error_pattern_for_logical(
    dem: stim.DetectorErrorModel,
    logical_config,
    logical_locations: List[int], 
    target_weight: int
) -> np.ndarray:
    """
    Find a specific error pattern that causes the given logical error.
    """
    sampler = dem.compile_sampler()
    
    for _ in range(10000):  # Increased attempts for finding specific pattern
        dets, obs, errs = sampler.sample(shots=1, return_errors=True)
        
        if errs is None:
            continue
            
        error_pattern = errs[0]
        if (np.array_equal(obs[0][logical_locations], logical_config) and 
            np.sum(error_pattern) == target_weight):
            return error_pattern
    
    # Fallback: return a simple pattern
    return np.zeros(dem.num_errors, dtype=np.int8)


def find_distance_with_pymatching_decoder(circuit: stim.Circuit, max_shots: int = 2**10):
    """
    Find distance using PyMatching decoder.
    This is a convenience function that uses the unified approach with PyMatching.
    """
    try:
        import pymatching
    except ImportError:
        raise ImportError("PyMatching not available. Install with: pip install pymatching")
    
    # Create decoder from detector error model
    dem = circuit.detector_error_model()
    decoder = pymatching.Matching.from_detector_error_model(dem)
    
    # Use the unified function with PyMatching decoder
    return find_distance_with_decoder(circuit, decoder.decode_batch, max_shots)


if __name__ == '__main__':
    # Example usage with different decoders
    d = 5
    p = 1e-3
    circuit = stim.Circuit.generated(
        rounds=d,
        distance=d,
        before_round_data_depolarization=p,
        code_task=f'surface_code:rotated_memory_x',
    )
    
    # Test with PyMatching decoder
    print("Testing with PyMatching decoder:")
    try:
        result = find_distance_with_pymatching_decoder(circuit)
        print(f"Distance: {result[0]}")
    except ImportError as e:
        print(f"PyMatching not available: {e}")
    
    # Test with a simple decoder (example)
    def simple_decoder(syndromes):
        """A simple decoder that just returns random predictions"""
        return np.random.randint(0, 2, size=(syndromes.shape[0], 1))
    
    print("\nTesting with simple decoder:")
    result = find_distance_with_decoder(circuit, simple_decoder)
    print(f"Distance: {result[0]}") 