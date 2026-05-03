from tests.test_training_smoke import test_cpu_quick_train, test_mps_quick_train_if_available


def main():
    print('Running CPU quick train test...')
    test_cpu_quick_train()
    print('CPU quick train test passed')

    print('Running MPS quick train test (device fallback to CPU if MPS unavailable)...')
    test_mps_quick_train_if_available()
    print('MPS quick train test passed')


if __name__ == '__main__':
    main()
