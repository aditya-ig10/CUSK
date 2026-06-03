from tests.test_training_smoke import (
    test_cpu_quick_train,
    test_mps_quick_train_if_available,
    test_no_overfitting,
    test_no_underfitting,
    test_learning_curve_improvement,
    test_accuracy_increasing
)


def main():
    print('=' * 60)
    print('SMOKE TESTS: Quick validation of training pipeline')
    print('=' * 60)
    
    print('\n1. Testing CPU quick train...')
    test_cpu_quick_train()
    print('   ✓ CPU quick train test passed')

    print('\n2. Testing MPS quick train (device fallback to CPU if MPS unavailable)...')
    test_mps_quick_train_if_available()
    print('   ✓ MPS quick train test passed')
    
    print('\n3. Testing no overfitting...')
    test_no_overfitting()
    
    print('\n4. Testing no underfitting...')
    test_no_underfitting()
    
    print('\n5. Testing learning curve improvement...')
    test_learning_curve_improvement()
    
    print('\n6. Testing accuracy increase over epochs...')
    test_accuracy_increasing()
    
    print('\n' + '=' * 60)
    print('ALL TESTS PASSED!')
    print('=' * 60)


if __name__ == '__main__':
    main()

