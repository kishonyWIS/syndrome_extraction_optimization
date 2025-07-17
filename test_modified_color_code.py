#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from color_code_modified import ColorCode

def test_modified_color_code():
    """Test the modified color code with correlated errors."""
    
    # Example correlated errors dictionary
    # Each key is an ancilla qubit name, each value is a list of sets of offset directions (0-5)
    # Offset directions: 0: (-2, 1), 1: (2, 1), 2: (4, 0), 3: (2, -1), 4: (-2, -1), 5: (-4, 0)
    correlated_errors = {
        "2-1-Z": [
            {0, 1, 2},  # Correlated error on data qubits at offsets 0, 1, 2 from ancilla (2,1,Z)
            {3, 4, 5}   # Correlated error on data qubits at offsets 3, 4, 5 from ancilla (2,1,Z)
        ],
        "4-2-X": [
            {0, 2, 4}   # Correlated error on data qubits at offsets 0, 2, 4 from ancilla (4,2,X)
        ]
    }
    
    try:
        # Create a modified color code with d=5, 2 rounds (hardcoded), and correlated errors
        colorcode = ColorCode(
            d=5,
            circuit_type="tri",
            cnot_schedule="tri_optimal",
            correlated_errors=correlated_errors,
            p_bitflip=0.1,  # 10% probability of bit-flip errors on all data qubits
            p_correlated=0.05  # 5% probability of correlated errors
        )
        
        print("✓ Successfully created modified ColorCode")
        print(f"  - Distance: {colorcode.d}")
        print(f"  - Rounds: {colorcode.rounds} (should be 2)")
        print(f"  - Circuit type: {colorcode.circuit_type}")
        print(f"  - Number of qubits: {colorcode.tanner_graph.vcount()}")
        print(f"  - Correlated errors: {len(colorcode.correlated_errors)} ancilla qubits")
        print(f"  - Bit-flip probability: {colorcode.p_bitflip}")
        print(f"  - Correlated error probability: {colorcode.p_correlated}")
        
        # Test circuit generation
        circuit = colorcode.circuit
        print(f"  - Circuit compiled successfully with {len(circuit)} instructions")
        
        # Check that we have 2 rounds
        assert colorcode.rounds == 2, f"Expected 2 rounds, got {colorcode.rounds}"
        print("✓ Rounds correctly hardcoded to 2")
        
        # Check that correlated errors are stored
        assert len(colorcode.correlated_errors) == 2, f"Expected 2 ancilla qubits, got {len(colorcode.correlated_errors)}"
        print("✓ Correlated errors correctly stored")
        
        # Search for undetectable logical errors
        print("\nSearching for undetectable logical errors...")
        undetectable_errors = circuit.search_for_undetectable_logical_errors(
            dont_explore_detection_event_sets_with_size_above=9999,
            dont_explore_edges_with_degree_above=9999,
            dont_explore_edges_increasing_symptom_degree=False,
            canonicalize_circuit_errors=False
        )
        
        print(f"  - Found {len(undetectable_errors)} undetectable logical error(s)")
        
        if undetectable_errors:
            print("  - Undetectable logical errors:")
            for i, error in enumerate(undetectable_errors):
                print(f"    Error {i+1}: {error}")
        else:
            print("  - No undetectable logical errors found")
        
        print("✓ Undetectable logical error search completed")
        
        print("\n✓ All tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_modified_color_code()
    sys.exit(0 if success else 1) 